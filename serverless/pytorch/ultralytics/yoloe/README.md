# YOLOE detector for CVAT

This function uses `yoloe-26l-seg.pt` with task labels as text prompts and returns rectangle annotations through CVAT's detector interface.

Start CVAT with the serverless component:

```powershell
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
```

Deploy the function from the repository root with `nuctl`:

```powershell
nuctl deploy pytorch-ultralytics-yoloe --project-name cvat --path serverless/pytorch/ultralytics/yoloe/nuclio --platform local
```

The NVIDIA Container Toolkit and WSL2 GPU support must be available to Docker Desktop. The function appears in the existing CVAT AI Tools detector list as `YOLOE Auto Label`. Its prompt list is populated from the current task labels, and the detector result is mapped back to CVAT label names before annotations are created.
