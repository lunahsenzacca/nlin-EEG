{ pkgs, lib, config, inputs, ... }:

{
  languages.python.enable = true;
  languages.python.version = "3.12.2";
  languages.python.venv.enable = true;
  languages.python.poetry.enable = true;
  languages.python.poetry.install.enable = true;
}
