using Microsoft.EntityFrameworkCore;
using MLOpsDashboard.Infrastructure.Data;
using MLOpsDashboard.Web.Components;
using MLOpsDashboard.Web.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

// Add health checks
builder.Services.AddHealthChecks()
    .AddNpgSql(builder.Configuration.GetConnectionString("DefaultConnection") ?? "");

// Add EF Core with PostgreSQL
builder.Services.AddDbContext<DashboardDbContext>(options =>
    options.UseNpgsql(builder.Configuration.GetConnectionString("DefaultConnection")));

// Add HTTP client for MLflow API
builder.Services.AddHttpClient<MlflowApiClient>(client =>
{
    client.BaseAddress = new Uri(builder.Configuration["Mlflow:BaseUrl"] ?? "http://mlflow:5000");
    client.Timeout = TimeSpan.FromSeconds(30);
});

// Add DAG generator service
builder.Services.AddScoped<DagGeneratorService>();

// Add template service
builder.Services.AddScoped<TemplateService>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
}
app.UseStatusCodePagesWithReExecute("/not-found");
app.UseAntiforgery();

// Health check endpoint
app.MapHealthChecks("/health");

app.UseStaticFiles();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

// Apply database migrations on startup
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<DashboardDbContext>();
    try
    {
        db.Database.Migrate();
    }
    catch (Exception ex)
    {
        var logger = scope.ServiceProvider.GetRequiredService<ILogger<Program>>();
        logger.LogWarning(ex, "Database migration failed - tables may already exist from init script");
    }
}

// Discover experiments from filesystem configs not yet tracked in the database
using (var scope = app.Services.CreateScope())
{
    var templateService = scope.ServiceProvider.GetRequiredService<TemplateService>();
    var config = scope.ServiceProvider.GetRequiredService<IConfiguration>();
    var logger = scope.ServiceProvider.GetRequiredService<ILogger<Program>>();
    var dagsPath = config["DagOutput:Path"] ?? "/app/dags";

    try
    {
        var count = await templateService.DiscoverExperimentsAsync(dagsPath);
        if (count > 0)
            logger.LogInformation("Experiment discovery complete: {Count} new experiment(s) registered", count);
    }
    catch (Exception ex)
    {
        logger.LogWarning(ex, "Experiment discovery failed - dashboard will continue without discovered experiments");
    }
}

app.Run();
