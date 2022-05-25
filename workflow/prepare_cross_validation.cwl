#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: Workflow

requirements:
  ScatterFeatureRequirement: { }

inputs:
  docker_image: string
  raw_data: File

outputs: []

steps:
  prepare:
    in:
      docker_image: docker_image
      raw_data: raw_data
    out:
      - cross_validation_configs
    run:
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
  run_cv:
    scatter: cross_validation_config
    in:
      cross_validation_config: prepare/cross_validation_configs
    out: []
    run:
      class: CommandLineTool
      arguments:
        - docker
        - run
        - "-v"
        - $(runtime.outdir):/data
        - $(inputs.docker_image)
        - grid_search
        - "--data-dir"
        - /data/k_fold/jobs/data
        - "--config"
        - /data/k_fold/setup/config/melanoma/config_1.cfg
        - "--output-dir"
        - /data/k_fold/setup/outputs/1
      inputs:
        cross_validation_config: File
      outputs: []
