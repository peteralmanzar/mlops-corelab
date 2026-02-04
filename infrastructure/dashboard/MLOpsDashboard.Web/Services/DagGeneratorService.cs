using System.Text.Json;
using MLOpsDashboard.Core.Models;
using MLOpsDashboard.Core.Models.ConfigSections;

namespace MLOpsDashboard.Web.Services;

/// <summary>
/// Service for generating experiment DAG files.
/// Creates Python DAG files and config.json for each experiment.
/// </summary>
public class DagGeneratorService
{
    private readonly IConfiguration _configuration;
    private readonly ILogger<DagGeneratorService> _logger;
    private readonly string _outputPath;

    public DagGeneratorService(IConfiguration configuration, ILogger<DagGeneratorService> logger)
    {
        _configuration = configuration;
        _logger = logger;
        _outputPath = configuration["DagOutput:Path"] ?? "/app/dags";
    }

    /// <summary>
    /// Generate all DAG files for an experiment.
    /// DAGs are placed directly in the dags folder for Airflow discovery.
    /// Config files are placed in configs/{experimentName}/ subfolder.
    /// </summary>
    public async Task<GenerationResult> GenerateExperimentAsync(
        string experimentName,
        ExperimentConfig config)
    {
        var result = new GenerationResult
        {
            ExperimentName = experimentName,
            GeneratedFiles = new List<string>()
        };

        try
        {
            // Create configs directory for this experiment
            var configDir = Path.Combine(_outputPath, "configs", experimentName);
            Directory.CreateDirectory(configDir);

            // Generate config.json in configs subfolder
            var configPath = Path.Combine(configDir, "config.json");
            var configJson = JsonSerializer.Serialize(config, new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = null // Keep original casing
            });
            await File.WriteAllTextAsync(configPath, configJson);
            result.GeneratedFiles.Add(configPath);

            // Generate DAG files directly in dags folder (not subfolder)
            result.GeneratedFiles.Add(await GenerateDataDagAsync(_outputPath, experimentName));
            result.GeneratedFiles.Add(await GeneratePreprocessDagAsync(_outputPath, experimentName));
            result.GeneratedFiles.Add(await GenerateModelDagAsync(_outputPath, experimentName));
            result.GeneratedFiles.Add(await GeneratePromoteDagAsync(_outputPath, experimentName));

            result.Success = true;
            result.GeneratedPath = _outputPath;

            _logger.LogInformation("Generated experiment {Name} DAGs at {Path}", experimentName, _outputPath);
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.ErrorMessage = ex.Message;
            _logger.LogError(ex, "Failed to generate experiment {Name}", experimentName);
        }

        return result;
    }

    private async Task<string> GenerateDataDagAsync(string dagsPath, string experimentName)
    {
        var dagPath = Path.Combine(dagsPath, $"{experimentName}_01_dag_data.py");
        var content = $"\"\"\"\n{experimentName}_01_dag_data - Data Pipeline DAG\n\nAuto-generated experiment DAG for {experimentName}.\nUses the template factory pattern from templates/base_01_data.py.\n\"\"\"\nimport sys\nfrom pathlib import Path\n\n# Explicit airflow import for DAG discovery (required by Airflow's safe mode)\nfrom airflow import DAG  # noqa: F401\n\n# Add templates to path\ndags_dir = Path(__file__).resolve().parent\nsys.path.insert(0, str(dags_dir))\n\nfrom templates.base_01_data import create_data_dag\n\n# Create the DAG using the factory\ndag = create_data_dag(\n    experiment_name=\"{experimentName}\",\n    config_path=str(dags_dir / \"configs\" / \"{experimentName}\" / \"config.json\")\n)\n";
        await File.WriteAllTextAsync(dagPath, content);
        return dagPath;
    }

    private async Task<string> GeneratePreprocessDagAsync(string dagsPath, string experimentName)
    {
        var dagPath = Path.Combine(dagsPath, $"{experimentName}_02_dag_preprocess.py");
        var content = $"\"\"\"\n{experimentName}_02_dag_preprocess - Preprocessing Pipeline DAG\n\nAuto-generated experiment DAG for {experimentName}.\nUses the template factory pattern from templates/base_02_preprocess.py.\n\"\"\"\nimport sys\nfrom pathlib import Path\n\n# Explicit airflow import for DAG discovery (required by Airflow's safe mode)\nfrom airflow import DAG  # noqa: F401\n\n# Add templates to path\ndags_dir = Path(__file__).resolve().parent\nsys.path.insert(0, str(dags_dir))\n\nfrom templates.base_02_preprocess import create_preprocess_dag\n\n# Create the DAG using the factory\ndag = create_preprocess_dag(\n    experiment_name=\"{experimentName}\",\n    config_path=str(dags_dir / \"configs\" / \"{experimentName}\" / \"config.json\")\n)\n";
        await File.WriteAllTextAsync(dagPath, content);
        return dagPath;
    }

    private async Task<string> GenerateModelDagAsync(string dagsPath, string experimentName)
    {
        var dagPath = Path.Combine(dagsPath, $"{experimentName}_03_dag_model.py");
        var content = $"\"\"\"\n{experimentName}_03_dag_model - Model Training DAG\n\nAuto-generated experiment DAG for {experimentName}.\nUses the template factory pattern from templates/base_03_model.py.\n\"\"\"\nimport sys\nfrom pathlib import Path\n\n# Explicit airflow import for DAG discovery (required by Airflow's safe mode)\nfrom airflow import DAG  # noqa: F401\n\n# Add templates to path\ndags_dir = Path(__file__).resolve().parent\nsys.path.insert(0, str(dags_dir))\n\nfrom templates.base_03_model import create_model_dag\n\n# Create the DAG using the factory\ndag = create_model_dag(\n    experiment_name=\"{experimentName}\",\n    config_path=str(dags_dir / \"configs\" / \"{experimentName}\" / \"config.json\")\n)\n";
        await File.WriteAllTextAsync(dagPath, content);
        return dagPath;
    }

    private async Task<string> GeneratePromoteDagAsync(string dagsPath, string experimentName)
    {
        var dagPath = Path.Combine(dagsPath, $"{experimentName}_04_dag_promote.py");
        var content = $"\"\"\"\n{experimentName}_04_dag_promote - Model Promotion DAG\n\nAuto-generated experiment DAG for {experimentName}.\nUses the template factory pattern from templates/base_04_promote.py.\n\nNote: Model reload is handled by 00_dag_mlflow_watcher which monitors\nMLflow Model Registry for champion alias changes across ALL experiments.\n\"\"\"\nimport sys\nfrom pathlib import Path\n\n# Explicit airflow import for DAG discovery (required by Airflow's safe mode)\nfrom airflow import DAG  # noqa: F401\n\n# Add templates to path\ndags_dir = Path(__file__).resolve().parent\nsys.path.insert(0, str(dags_dir))\n\nfrom templates.base_04_promote import create_promote_dag\n\n# Create the DAG using the factory\ndag = create_promote_dag(\n    experiment_name=\"{experimentName}\",\n    config_path=str(dags_dir / \"configs\" / \"{experimentName}\" / \"config.json\")\n)\n";
        await File.WriteAllTextAsync(dagPath, content);
        return dagPath;
    }

    /// <summary>
    /// Delete an experiment's generated files.
    /// Removes DAG files from root and config folder.
    /// </summary>
    public Task<bool> DeleteExperimentAsync(string experimentName)
    {
        try
        {
            var deleted = false;

            // Delete DAG files from root
            var dagFiles = new[]
            {
                Path.Combine(_outputPath, $"{experimentName}_01_dag_data.py"),
                Path.Combine(_outputPath, $"{experimentName}_02_dag_preprocess.py"),
                Path.Combine(_outputPath, $"{experimentName}_03_dag_model.py"),
                Path.Combine(_outputPath, $"{experimentName}_04_dag_promote.py")
            };

            foreach (var dagFile in dagFiles)
            {
                if (File.Exists(dagFile))
                {
                    File.Delete(dagFile);
                    deleted = true;
                }
            }

            // Delete config folder
            var configDir = Path.Combine(_outputPath, "configs", experimentName);
            if (Directory.Exists(configDir))
            {
                Directory.Delete(configDir, recursive: true);
                deleted = true;
            }

            if (deleted)
            {
                _logger.LogInformation("Deleted experiment files for {Name}", experimentName);
            }

            return Task.FromResult(deleted);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to delete experiment {Name}", experimentName);
            return Task.FromResult(false);
        }
    }
}

public class GenerationResult
{
    public required string ExperimentName { get; set; }
    public bool Success { get; set; }
    public string? GeneratedPath { get; set; }
    public List<string> GeneratedFiles { get; set; } = new();
    public string? ErrorMessage { get; set; }
}
