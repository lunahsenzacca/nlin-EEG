#!/usr/bin/env bash

kitty -d ~/nlin-EEG -o allow_remote_control=yes -o enabled_layouts=fat:bias=90 -e devenv shell --profile dev nvim
