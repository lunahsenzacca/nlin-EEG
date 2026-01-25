{ pkgs, lib, config, inputs, ... }:

{

  env.GREET = ''
    Let's try doing something today!
  '';

  scripts.hello.exec = ''
    echo $GREET
  '';

  enterShell = ''
    hello
    export SHELL=${pkgs.bashInteractive}/bin/bash
    source .aliases
  '';

  profiles = {
    dev.module = {
      enterShell = ''
        kitten @ new-window --keep-focus=yes devenv shell
        nvim
      '';
    };
  };

  packages = with pkgs; [

    bashInteractive
    kitty
    micro

    gcc
    libz
    python313Packages.qtpy
    python313Packages.pyqt6
    python313Packages.pyside6
    python313Packages.pyqtgraph

  ];

  languages.python.enable = true;
  languages.python.version = "3.13";
  languages.python.venv.enable = true;
  languages.python.poetry.enable = true;
  languages.python.poetry.install.enable = true;
}
