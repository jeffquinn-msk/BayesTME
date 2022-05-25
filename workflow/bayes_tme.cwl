#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: Workflow

requirements:
  ScatterFeatureRequirement: { }
  SubworkflowFeatureRequirement: { }

inputs:
  data: File

steps:
  filter_bleed: { }
  prepare_grid_search: { }
  grid_search: { }
  deconvolve: { }
  spatial_expression: { }