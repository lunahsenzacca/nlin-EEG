{ pkgs, lib, config, inputs, ... }:

{
  packages = with pkgs; [
    gcc
    libz
    python312Packages.qtpy
    python312Packages.pyqt6
    python312Packages.pyside6
    python312Packages.pyqtgraph
  ];
  languages.python.enable = true;
  languages.python.version = "3.13";
  languages.python.venv.enable = true;
  languages.python.poetry.enable = true;
  languages.python.poetry.install.enable = true;
}
