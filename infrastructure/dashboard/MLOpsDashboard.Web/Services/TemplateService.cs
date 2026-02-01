using Microsoft.EntityFrameworkCore;
using MLOpsDashboard.Core.Models;
using MLOpsDashboard.Infrastructure.Data;

namespace MLOpsDashboard.Web.Services;

/// <summary>
/// Service for managing experiment templates.
/// </summary>
public class TemplateService
{
    private readonly DashboardDbContext _context;
    private readonly ILogger<TemplateService> _logger;

    public TemplateService(DashboardDbContext context, ILogger<TemplateService> logger)
    {
        _context = context;
        _logger = logger;
    }

    /// <summary>
    /// Get all experiment templates.
    /// </summary>
    public async Task<List<ExperimentTemplate>> GetAllTemplatesAsync()
    {
        return await _context.ExperimentTemplates
            .OrderByDescending(t => t.UpdatedAt)
            .ToListAsync();
    }

    /// <summary>
    /// Get a template by ID.
    /// </summary>
    public async Task<ExperimentTemplate?> GetTemplateByIdAsync(Guid id)
    {
        return await _context.ExperimentTemplates
            .Include(t => t.GeneratedExperiments)
            .FirstOrDefaultAsync(t => t.Id == id);
    }

    /// <summary>
    /// Get a template by name.
    /// </summary>
    public async Task<ExperimentTemplate?> GetTemplateByNameAsync(string name)
    {
        return await _context.ExperimentTemplates
            .FirstOrDefaultAsync(t => t.Name == name);
    }

    /// <summary>
    /// Create a new template.
    /// </summary>
    public async Task<ExperimentTemplate> CreateTemplateAsync(ExperimentTemplate template)
    {
        template.CreatedAt = DateTime.UtcNow;
        template.UpdatedAt = DateTime.UtcNow;

        _context.ExperimentTemplates.Add(template);
        await _context.SaveChangesAsync();

        _logger.LogInformation("Created template: {Name}", template.Name);
        return template;
    }

    /// <summary>
    /// Update an existing template.
    /// </summary>
    public async Task<ExperimentTemplate> UpdateTemplateAsync(ExperimentTemplate template)
    {
        template.UpdatedAt = DateTime.UtcNow;

        _context.ExperimentTemplates.Update(template);
        await _context.SaveChangesAsync();

        _logger.LogInformation("Updated template: {Name}", template.Name);
        return template;
    }

    /// <summary>
    /// Delete a template (if no experiments generated from it).
    /// </summary>
    public async Task<bool> DeleteTemplateAsync(Guid id)
    {
        var template = await _context.ExperimentTemplates
            .Include(t => t.GeneratedExperiments)
            .FirstOrDefaultAsync(t => t.Id == id);

        if (template == null)
            return false;

        if (template.GeneratedExperiments.Any())
        {
            _logger.LogWarning("Cannot delete template {Name} - has generated experiments", template.Name);
            return false;
        }

        _context.ExperimentTemplates.Remove(template);
        await _context.SaveChangesAsync();

        _logger.LogInformation("Deleted template: {Name}", template.Name);
        return true;
    }

    /// <summary>
    /// Get all generated experiments.
    /// </summary>
    public async Task<List<GeneratedExperiment>> GetAllExperimentsAsync()
    {
        return await _context.GeneratedExperiments
            .Include(e => e.Template)
            .OrderByDescending(e => e.CreatedAt)
            .ToListAsync();
    }

    /// <summary>
    /// Get a generated experiment by ID.
    /// </summary>
    public async Task<GeneratedExperiment?> GetExperimentByIdAsync(Guid id)
    {
        return await _context.GeneratedExperiments
            .Include(e => e.Template)
            .FirstOrDefaultAsync(e => e.Id == id);
    }

    /// <summary>
    /// Create a generated experiment record.
    /// </summary>
    public async Task<GeneratedExperiment> CreateExperimentAsync(GeneratedExperiment experiment)
    {
        experiment.CreatedAt = DateTime.UtcNow;
        experiment.UpdatedAt = DateTime.UtcNow;

        _context.GeneratedExperiments.Add(experiment);
        await _context.SaveChangesAsync();

        _logger.LogInformation("Created experiment: {Name}", experiment.Name);
        return experiment;
    }

    /// <summary>
    /// Update experiment status.
    /// </summary>
    public async Task UpdateExperimentStatusAsync(Guid id, string status, string? errorMessage = null)
    {
        var experiment = await _context.GeneratedExperiments.FindAsync(id);
        if (experiment != null)
        {
            experiment.Status = status;
            experiment.ErrorMessage = errorMessage;
            experiment.UpdatedAt = DateTime.UtcNow;
            await _context.SaveChangesAsync();
        }
    }

    /// <summary>
    /// Update a generated experiment.
    /// </summary>
    public async Task<GeneratedExperiment> UpdateExperimentAsync(GeneratedExperiment experiment)
    {
        experiment.UpdatedAt = DateTime.UtcNow;
        _context.GeneratedExperiments.Update(experiment);
        await _context.SaveChangesAsync();
        _logger.LogInformation("Updated experiment: {Name}", experiment.Name);
        return experiment;
    }
}
