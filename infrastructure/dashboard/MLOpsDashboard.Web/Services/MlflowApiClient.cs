using System.Text.Json;

namespace MLOpsDashboard.Web.Services;

/// <summary>
/// HTTP client for MLflow REST API.
/// Used to fetch experiment results and model registry information.
/// </summary>
public class MlflowApiClient
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<MlflowApiClient> _logger;

    public MlflowApiClient(HttpClient httpClient, ILogger<MlflowApiClient> logger)
    {
        _httpClient = httpClient;
        _logger = logger;
    }

    /// <summary>
    /// Get all experiments from MLflow.
    /// </summary>
    public async Task<List<MlflowExperiment>> GetExperimentsAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync("/api/2.0/mlflow/experiments/search");
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            var result = JsonSerializer.Deserialize<MlflowExperimentsResponse>(json);

            return result?.Experiments ?? new List<MlflowExperiment>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to fetch experiments from MLflow");
            return new List<MlflowExperiment>();
        }
    }

    /// <summary>
    /// Get registered models from MLflow Model Registry.
    /// </summary>
    public async Task<List<MlflowRegisteredModel>> GetRegisteredModelsAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync("/api/2.0/mlflow/registered-models/search");
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            var result = JsonSerializer.Deserialize<MlflowRegisteredModelsResponse>(json);

            return result?.RegisteredModels ?? new List<MlflowRegisteredModel>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to fetch registered models from MLflow");
            return new List<MlflowRegisteredModel>();
        }
    }

    /// <summary>
    /// Get runs for a specific experiment.
    /// </summary>
    public async Task<List<MlflowRun>> GetRunsAsync(string experimentId, int maxResults = 100)
    {
        try
        {
            var request = new
            {
                experiment_ids = new[] { experimentId },
                max_results = maxResults
            };

            var response = await _httpClient.PostAsJsonAsync("/api/2.0/mlflow/runs/search", request);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            var result = JsonSerializer.Deserialize<MlflowRunsResponse>(json);

            return result?.Runs ?? new List<MlflowRun>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to fetch runs for experiment {ExperimentId}", experimentId);
            return new List<MlflowRun>();
        }
    }
}

// MLflow API response models
public class MlflowExperimentsResponse
{
    public List<MlflowExperiment>? Experiments { get; set; }
}

public class MlflowExperiment
{
    public string? ExperimentId { get; set; }
    public string? Name { get; set; }
    public string? LifecycleStage { get; set; }
}

public class MlflowRegisteredModelsResponse
{
    public List<MlflowRegisteredModel>? RegisteredModels { get; set; }
}

public class MlflowRegisteredModel
{
    public string? Name { get; set; }
    public List<MlflowModelVersion>? LatestVersions { get; set; }
}

public class MlflowModelVersion
{
    public string? Version { get; set; }
    public string? CurrentStage { get; set; }
    public List<string>? Aliases { get; set; }
}

public class MlflowRunsResponse
{
    public List<MlflowRun>? Runs { get; set; }
}

public class MlflowRun
{
    public MlflowRunInfo? Info { get; set; }
    public MlflowRunData? Data { get; set; }
}

public class MlflowRunInfo
{
    public string? RunId { get; set; }
    public string? RunName { get; set; }
    public string? Status { get; set; }
    public long? StartTime { get; set; }
    public long? EndTime { get; set; }
}

public class MlflowRunData
{
    public List<MlflowMetric>? Metrics { get; set; }
    public List<MlflowParam>? Params { get; set; }
}

public class MlflowMetric
{
    public string? Key { get; set; }
    public double? Value { get; set; }
}

public class MlflowParam
{
    public string? Key { get; set; }
    public string? Value { get; set; }
}
