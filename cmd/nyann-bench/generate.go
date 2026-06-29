package main

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/neuralmagic/nyann-bench/pkg/analysis"
	"github.com/neuralmagic/nyann-bench/pkg/barrier"
	"github.com/neuralmagic/nyann-bench/pkg/config"
	"github.com/neuralmagic/nyann-bench/pkg/kube"
	promclient "github.com/neuralmagic/nyann-bench/pkg/prometheus"
	"github.com/neuralmagic/nyann-bench/pkg/recorder"
	"github.com/spf13/cobra"
)

func generateCmd() *cobra.Command {
	var (
		target        string
		model         string
		cfgInput      string
		outputDir     string
		workerID      int
		workersFlag   string
		metricsAddr   string
		prometheusURL string
		deployName    string
		profileRole   string
		profileAfter  int
		kubeFlags     kube.Flags
	)

	cmd := &cobra.Command{
		Use:   "generate",
		Short: "Generate load against an LLM inference endpoint",
		Long: `Generate load against an LLM inference endpoint.

Configure the workload via --config (JSON file, inline JSON, or Starlark .star file):

  nyann-bench generate --target http://localhost:8000/v1 --model my-model \
    --config '{"load":{"mode":"concurrent","concurrency":10,"duration":"60s"},"workload":{"type":"faker","isl":128,"osl":256}}'

  nyann-bench generate --target http://localhost:8000/v1 --config benchmark.json

  nyann-bench generate --config scenario.star

Starlark (.star) files provide full programmability — loops, functions,
conditionals, and per-stage workload/target overrides:

  scenario(
      stages = [stage("2m", concurrency=c) for c in range(10, 101, 10)],
      workload = workload("faker", isl=512, osl=1024),
  )

Load modes:
  concurrent  Fixed number of streams, each fires next request on completion (default)
  constant    Requests arrive at a fixed rate (evenly spaced)
  poisson     Requests arrive at a target rate with exponential inter-arrival times

Workload types:
  synthetic   Random word padding
  faker       Diverse generated prose (gofakeit)
  corpus      Sliding window over real text files
  gsm8k       GSM8K math problems with streaming eval`,
		RunE: func(cmd *cobra.Command, args []string) error {
			// Parse config early — needed to resolve --workers auto
			sc, err := config.Parse(cfgInput)
			if err != nil {
				return fmt.Errorf("config: %w", err)
			}

			workers, err := config.ResolveWorkers(workersFlag, sc.MaxConcurrency())
			if err != nil {
				return err
			}
			if workersFlag == "auto" {
				slog.Info("Auto-resolved workers", "workers", workers, "max_concurrency", sc.MaxConcurrency())
			}

			if kubeFlags.IsEnabled(cmd) {
				cfg, err := kubeFlags.ToConfig()
				if err != nil {
					return err
				}
				if workers > 1 {
					cfg.Workers = workers
				}
				containerArgs := kube.CollectArgs(cmd, []string{"generate"})
				containerArgs = append(containerArgs, "--metrics", ":9090")
				if cfg.Workers > 1 && !cmd.Flags().Changed("workers") {
					containerArgs = append(containerArgs, "--workers", strconv.Itoa(cfg.Workers))
				}
				return kube.Deploy(cfg, "generate", containerArgs)
			}

			ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
			defer cancel()

			// Auto-detect worker ID from K8s indexed Job
			if workerID == 0 {
				if idx, ok := os.LookupEnv("JOB_COMPLETION_INDEX"); ok {
					if v, err := strconv.Atoi(idx); err == nil {
						workerID = v
					}
				}
			}

			sc.Workers = workers
			sc.WorkerID = workerID

			// Configure barrier sync for multi-worker runs
			if workers > 1 {
				syncCfg := &config.SyncConfig{
					Workers: workers,
					Timeout: config.Duration(10 * time.Minute),
					Port:    8080,
				}
				if addr, ok := os.LookupEnv("BARRIER_ADDR"); ok {
					syncCfg.Addr = addr
				} else {
					syncCfg.Addr = "localhost"
				}
				sc.Sync = syncCfg
				sc.InsertImplicitBarrier()

				if workerID == 0 {
					srv := barrier.NewServer(workers, syncCfg.Port)
					go srv.ListenAndServe(ctx)
				}

				slog.Info("Sync enabled", "workers", workers, "addr", syncCfg.Addr, "port", syncCfg.Port)
			}

			// CLI flags override config-level target/model
			if sc.Target != "" && target == "http://localhost:8000/v1" {
				target = sc.Target
			}
			if sc.Model != "" && model == "" {
				model = sc.Model
			}

			var promClient *promclient.Client
			if prometheusURL != "" && deployName != "" {
				promClient = promclient.NewClient(prometheusURL)
				prefillDeploy := strings.Replace(deployName, "-decode", "-prefill", 1)
				slog.Info("Prometheus scrape targets",
					"url", prometheusURL,
					"decode_pods", deployName+".*",
					"prefill_pods", prefillDeploy+".*")
			}

			headerPrinted := false
			var collected []*analysis.ServerMetrics

			opts := scenarioOpts{
				Target:      target,
				Model:       model,
				Scenario:    sc,
				OutputDir:   outputDir,
				WorkerID:    workerID,
				MetricsAddr: metricsAddr,
				OnStageComplete: func(ts recorder.StageTimestamp, records []recorder.Record) {
					stages := analysis.ComputePerStage(records, []recorder.StageTimestamp{ts})
					if len(stages) == 0 {
						return
					}
					stage := stages[0]

					if promClient != nil {
						server := analysis.QueryStageServerMetrics(promClient, ts, deployName)
						stage.Server = server
						collected = append(collected, server)
					} else {
						collected = append(collected, nil)
					}

					if !headerPrinted {
						fmt.Fprint(os.Stderr, analysis.FormatStageHeader(stage.Server != nil))
						headerPrinted = true
					}
					fmt.Fprint(os.Stderr, analysis.FormatStageRow(stage))
				},
			}

			if profileRole != "" {
				if deployName == "" {
					return fmt.Errorf("--profile requires --deploy-name")
				}
				if profileRole != "decode" && profileRole != "prefill" {
					return fmt.Errorf("--profile must be 'decode' or 'prefill'")
				}
				podPrefix := deployName
				port := 8200
				if profileRole == "prefill" {
					podPrefix = strings.Replace(deployName, "-decode", "-prefill", 1)
					port = 8000
				}
				profileURLs, err := discoverPodURLs(podPrefix, port)
				if err != nil {
					return fmt.Errorf("discovering %s pods: %w", profileRole, err)
				}
				if len(profileURLs) == 0 {
					return fmt.Errorf("no running %s pods found matching %s", profileRole, podPrefix)
				}
				slog.Info("Per-stage profiling enabled", "role", profileRole, "targets", profileURLs, "after", profileAfter)
				waitThreshold := profileAfter
				var profileStarted sync.WaitGroup
				opts.OnStageProfileStart = func(stage, concurrency int, recordCount func() int) {
					profileStarted.Add(1)
					go func() {
						defer profileStarted.Done()
						if waitThreshold > 0 {
							baseline := recordCount()
							target := baseline + waitThreshold
							slog.Info("Waiting for requests before profiling", "stage", stage, "need", waitThreshold)
							for recordCount() < target {
								time.Sleep(100 * time.Millisecond)
							}
						}
						slog.Info("Starting profile", "stage", stage, "concurrency", concurrency)
						postProfileAction(profileURLs, "/start_profile")
					}()
				}
				opts.OnStageProfileStop = func(stage, concurrency int) {
					profileStarted.Wait()
					slog.Info("Stopping profile", "stage", stage, "concurrency", concurrency)
					postProfileAction(profileURLs, "/stop_profile")
				}
			}

			summary, err := runScenario(ctx, cancel, opts)
			if err != nil {
				return err
			}

			// Requery Prometheus for all stages now that the benchmark is done.
			if promClient != nil && summary.Timestamps != nil {
				for i, ts := range summary.Timestamps.Stages {
					if i < len(summary.Stages) {
						summary.Stages[i].Server = analysis.QueryStageServerMetrics(promClient, ts, deployName)
					}
				}
			}

			if summary.TotalRequests > 0 {
				fmt.Fprint(os.Stderr, "\n")
				if len(summary.Stages) > 0 {
					hasServer := promClient != nil
					fmt.Fprint(os.Stderr, analysis.FormatStageHeader(hasServer))
					for _, s := range summary.Stages {
						fmt.Fprint(os.Stderr, analysis.FormatStageRow(s))
					}
					fmt.Fprint(os.Stderr, "\n")
				}
				fmt.Fprint(os.Stderr, analysis.FormatSummary(summary))

				jsonOut, err := json.MarshalIndent(summary, "", "  ")
				if err != nil {
					return fmt.Errorf("marshalling summary: %w", err)
				}
				fmt.Println(string(jsonOut))
			}

			return nil
		},
	}

	cmd.Flags().StringVar(&target, "target", "http://localhost:8000/v1", "Target endpoint base URL")
	cmd.Flags().StringVar(&model, "model", "", "Model name for requests")
	cmd.Flags().StringVar(&cfgInput, "config", "{}", "Workload config (JSON file, inline JSON, or .star file)")
	cmd.Flags().StringVar(&outputDir, "output-dir", "", "Directory for JSONL + timestamp files (omit for stdout-only)")
	cmd.Flags().IntVar(&workerID, "worker-id", 0, "Worker identifier (for multi-container runs)")
	cmd.Flags().StringVar(&workersFlag, "workers", "1", `Number of workers: integer or "auto" (auto = ceil(max_concurrency/1024))`)
	cmd.Flags().StringVar(&metricsAddr, "metrics", "", "Prometheus metrics listen address (e.g. :9090)")
	cmd.Flags().StringVar(&prometheusURL, "prometheus-url", "", "Prometheus server URL for querying server-side vLLM metrics (e.g. http://prometheus:9090)")
	cmd.Flags().StringVar(&deployName, "deploy-name", "", "Deployment name prefix for Prometheus pod label filtering (e.g. my-deploy)")
	cmd.Flags().StringVar(&profileRole, "profile", "", "Enable per-stage torch profiling on vLLM pods (decode or prefill). Requires --deploy-name.")
	cmd.Flags().IntVar(&profileAfter, "profile-after", 0, "Wait for at least N completed requests in the stage before starting the profile")

	kube.RegisterFlags(cmd, &kubeFlags)

	return cmd
}

func postProfileAction(urls []string, path string) {
	var wg sync.WaitGroup
	for _, u := range urls {
		wg.Add(1)
		go func(baseURL string) {
			defer wg.Done()
			url := strings.TrimRight(baseURL, "/") + path
			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()
			req, err := http.NewRequestWithContext(ctx, "POST", url, nil)
			if err != nil {
				slog.Warn("Profile request failed", "url", url, "error", err)
				return
			}
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				slog.Warn("Profile request failed", "url", url, "error", err)
				return
			}
			resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				slog.Warn("Profile request non-200", "url", url, "status", resp.StatusCode)
			}
		}(u)
	}
	wg.Wait()
}

func discoverPodURLs(podPrefix string, port int) ([]string, error) {
	token, err := os.ReadFile("/var/run/secrets/kubernetes.io/serviceaccount/token")
	if err != nil {
		return nil, fmt.Errorf("reading service account token: %w (not running in-cluster?)", err)
	}
	ns, err := os.ReadFile("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
	if err != nil {
		return nil, fmt.Errorf("reading namespace: %w", err)
	}

	apiURL := fmt.Sprintf("https://kubernetes.default.svc/api/v1/namespaces/%s/pods?fieldSelector=status.phase=Running", string(ns))

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+string(token))

	client := &http.Client{Transport: &http.Transport{TLSClientConfig: tlsConfigSkipVerify()}}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("querying K8s API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("K8s API returned %d: %s", resp.StatusCode, string(body))
	}

	var podList struct {
		Items []struct {
			Status struct {
				PodIP string `json:"podIP"`
				Phase string `json:"phase"`
			} `json:"status"`
			Metadata struct {
				Name string `json:"name"`
			} `json:"metadata"`
		} `json:"items"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&podList); err != nil {
		return nil, fmt.Errorf("decoding pod list: %w", err)
	}

	var urls []string
	for _, pod := range podList.Items {
		if strings.HasPrefix(pod.Metadata.Name, podPrefix) && pod.Status.PodIP != "" {
			urls = append(urls, fmt.Sprintf("http://%s:%d", pod.Status.PodIP, port))
		}
	}
	return urls, nil
}

func tlsConfigSkipVerify() *tls.Config {
	return &tls.Config{InsecureSkipVerify: true} //nolint:gosec // in-cluster K8s API
}
