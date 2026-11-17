package main

import (
	"context"
	"fmt"
	"os"

	"dagger.io/dagger"
)

// Start docker and run': cd dagger
//go run .

// pipeline runs the same smoke pipeline used on locally,
//  The host repository is mounted read-only at /project.
// We need to install the dev dependencies to run the tests and train the model.
// The artifacts are exported to the host at ../artifacts.
// The trained model is served in a container and tested with a curl command.
// remebmer to make the smoke test globally later !!! Dont forget -Champ !!!!

func pipeline(ctx context.Context, client *dagger.Client) error {
	source := client.Host().Directory("..")
	container := client.Container().
		From("python:3.13-slim").
		WithDirectory("/project", source).
		WithWorkdir("/project").
		WithExec([]string{"pip", "install", "--no-cache-dir", "-e", ".[dev,serve]"}).
		WithEnvVariable("VIAL_ARTIFACT_DIR", "/project/artifacts").
		WithExec([]string{"vial-pipeline", "test"}).
		WithExec([]string{"vial-pipeline", "train", "--mode", "smoke"}).
		WithExec([]string{"vial-pipeline", "validate"})

	if _, err := container.Stdout(ctx); err != nil {
		return fmt.Errorf("pipeline failed: %w", err)
	}

	artifact := container.Directory("/project/artifacts")
	if _, err := artifact.Export(ctx, "../artifacts"); err != nil {
		return fmt.Errorf("export smoke artifact: %w", err)
	}

	image := client.Container().
		From("python:3.13-slim").
		WithDirectory("/app", client.Host().Directory("..")).
		WithWorkdir("/app").
		WithExec([]string{"pip", "install", "--no-cache-dir", "-e", ".[serve]"}).
		WithEntrypoint([]string{
			"uvicorn", "vial_classifier.api:app",
			"--host", "0.0.0.0", "--port", "5001",
		}).
		WithExposedPort(5001)
	service := image.AsService()
	smoke := client.Container().
		From("curlimages/curl:8.10.1").
		WithServiceBinding("model", service).
		WithExec([]string{"curl", "--fail", "--silent", "http://model:5001/health"})
	output, err := smoke.Stdout(ctx)
	if err != nil {
		return fmt.Errorf("serving smoke test failed: %w", err)
	}
	if output == "" {
		return fmt.Errorf("serving smoke test returned an empty health response")
	}
	return nil
}

func main() {
	ctx := context.Background()
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stderr))
	if err != nil {
		fmt.Fprintf(os.Stderr, "connect to dagger: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	if err := pipeline(ctx, client); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
