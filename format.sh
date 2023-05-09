#!/bin/bash
black .
isort .
autoflake -ri --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables .
isort .
#mypy .
black .
