#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: CommandLineTool
arguments:
  - docker
  - run
  - "-v"
  - $(runtime.outdir):/data
  - "-v"
  - $(inputs.raw_data.path):/input
  - $(inputs.docker_image)
  - prepare_kfold
  - "--count-mat"
  - /input
  - "--data-dir"
  - /data
inputs:
  docker_image: string
  raw_data: File
outputs:
  cross_validation_configs:
    type: File[]
    outputBinding:
      glob: "$(runtime.outdir)/k_fold/setup/config/BayesTME/*.cfg"