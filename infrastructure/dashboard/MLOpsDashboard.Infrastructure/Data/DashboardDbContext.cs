using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Diagnostics;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;
using MLOpsDashboard.Core.Models;

namespace MLOpsDashboard.Infrastructure.Data;

/// <summary>
/// EF Core DbContext for the MLOps Dashboard.
/// Uses PostgreSQL as the backend database.
/// Tables are created by SQL init scripts, not EF migrations.
/// </summary>
public class DashboardDbContext : DbContext
{
    public DashboardDbContext(DbContextOptions<DashboardDbContext> options)
        : base(options)
    {
    }

    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
    {
        base.OnConfiguring(optionsBuilder);
        // Suppress pending model changes warning - we use SQL scripts for schema, not EF migrations
        optionsBuilder.ConfigureWarnings(w => w.Ignore(RelationalEventId.PendingModelChangesWarning));
    }

    public DbSet<ExperimentTemplate> ExperimentTemplates => Set<ExperimentTemplate>();
    public DbSet<GeneratedExperiment> GeneratedExperiments => Set<GeneratedExperiment>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder);

        // Value converter to ensure DateTime values are always UTC
        // PostgreSQL timestamptz columns require UTC, but EF Core reads them as Kind=Unspecified
        var dateTimeUtcConverter = new ValueConverter<DateTime, DateTime>(
            v => v.Kind == DateTimeKind.Unspecified ? DateTime.SpecifyKind(v, DateTimeKind.Utc) : v.ToUniversalTime(),
            v => DateTime.SpecifyKind(v, DateTimeKind.Utc));

        // Configure ExperimentTemplate
        modelBuilder.Entity<ExperimentTemplate>(entity =>
        {
            entity.ToTable("experiment_templates");

            entity.HasKey(e => e.Id);

            // Map to snake_case column names (PostgreSQL convention)
            entity.Property(e => e.Id).HasColumnName("id");
            entity.Property(e => e.Name).HasColumnName("name")
                .IsRequired()
                .HasMaxLength(255);
            entity.Property(e => e.Description).HasColumnName("description")
                .HasMaxLength(1000);
            entity.Property(e => e.TemplateType).HasColumnName("template_type")
                .IsRequired()
                .HasMaxLength(50)
                .HasDefaultValue("full_pipeline");
            entity.Property(e => e.DefaultConfig).HasColumnName("default_config")
                .HasColumnType("jsonb");
            entity.Property(e => e.CreatedAt).HasColumnName("created_at")
                .HasDefaultValueSql("CURRENT_TIMESTAMP")
                .HasConversion(dateTimeUtcConverter);
            entity.Property(e => e.UpdatedAt).HasColumnName("updated_at")
                .HasDefaultValueSql("CURRENT_TIMESTAMP")
                .HasConversion(dateTimeUtcConverter);

            entity.HasIndex(e => e.Name)
                .IsUnique();
        });

        // Configure GeneratedExperiment
        modelBuilder.Entity<GeneratedExperiment>(entity =>
        {
            entity.ToTable("generated_experiments");

            entity.HasKey(e => e.Id);

            // Map to snake_case column names (PostgreSQL convention)
            entity.Property(e => e.Id).HasColumnName("id");
            entity.Property(e => e.TemplateId).HasColumnName("template_id");
            entity.Property(e => e.Name).HasColumnName("name")
                .IsRequired()
                .HasMaxLength(255);
            entity.Property(e => e.ConfigSnapshot).HasColumnName("config_snapshot")
                .HasColumnType("jsonb");
            entity.Property(e => e.GeneratedPath).HasColumnName("generated_path")
                .HasMaxLength(500);
            entity.Property(e => e.Status).HasColumnName("status")
                .IsRequired()
                .HasMaxLength(50)
                .HasDefaultValue("generated");
            entity.Property(e => e.LastAirflowRunId).HasColumnName("last_airflow_run_id")
                .HasMaxLength(255);
            entity.Property(e => e.ErrorMessage).HasColumnName("error_message")
                .HasMaxLength(2000);
            entity.Property(e => e.CreatedAt).HasColumnName("created_at")
                .HasDefaultValueSql("CURRENT_TIMESTAMP")
                .HasConversion(dateTimeUtcConverter);
            entity.Property(e => e.UpdatedAt).HasColumnName("updated_at")
                .HasDefaultValueSql("CURRENT_TIMESTAMP")
                .HasConversion(dateTimeUtcConverter);

            entity.HasIndex(e => e.Name)
                .IsUnique();

            entity.HasOne(e => e.Template)
                .WithMany(t => t.GeneratedExperiments)
                .HasForeignKey(e => e.TemplateId)
                .OnDelete(DeleteBehavior.Restrict);
        });
    }
}
