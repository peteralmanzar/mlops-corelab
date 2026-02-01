# Build stage
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy solution and project files
COPY MLOpsDashboard.sln .
COPY MLOpsDashboard.Web/MLOpsDashboard.Web.csproj MLOpsDashboard.Web/
COPY MLOpsDashboard.Core/MLOpsDashboard.Core.csproj MLOpsDashboard.Core/
COPY MLOpsDashboard.Infrastructure/MLOpsDashboard.Infrastructure.csproj MLOpsDashboard.Infrastructure/

# Restore dependencies
RUN dotnet restore

# Copy source code
COPY . .

# Build and publish
RUN dotnet publish MLOpsDashboard.Web/MLOpsDashboard.Web.csproj -c Release -o /app/publish --no-restore

# Runtime stage
FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS runtime
WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy published app
COPY --from=build /app/publish .

# Create directory for generated DAGs
RUN mkdir -p /app/dags

# Expose port
EXPOSE 3001

# Set environment variables
ENV ASPNETCORE_URLS=http://+:3001
ENV ASPNETCORE_ENVIRONMENT=Production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3001/health || exit 1

# Run the application
ENTRYPOINT ["dotnet", "MLOpsDashboard.Web.dll"]
