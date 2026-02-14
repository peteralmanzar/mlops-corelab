namespace MLOpsDashboard.Web.Services;

/// <summary>
/// HTTP client for the Model Serve FastAPI service.
/// Used to construct export URLs for standalone model packaging.
/// </summary>
public class ModelServeApiClient
{
    private readonly IConfiguration _configuration;
    private readonly ILogger<ModelServeApiClient> _logger;

    public ModelServeApiClient(IConfiguration configuration, ILogger<ModelServeApiClient> logger)
    {
        _configuration = configuration;
        _logger = logger;
    }

    /// <summary>
    /// Get the external URL for exporting a model as a standalone Docker package.
    /// This URL is used by the browser to trigger a download directly from model-serve.
    /// </summary>
    public string GetExportUrl(string modelName)
    {
        var baseUrl = _configuration["ModelServe:ExternalBaseUrl"] ?? "http://localhost:8000";
        baseUrl = baseUrl.TrimEnd('/');
        return $"{baseUrl}/admin/export/{Uri.EscapeDataString(modelName)}";
    }
}
