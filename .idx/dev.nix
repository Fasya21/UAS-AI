{ pkgs, ... }: {
  channel = "stable-24.05";

  packages = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.nodejs_20
    pkgs.nodePackages.nodemon
  ];

  env = {};

  idx = {
    extensions = [];

    previews = {
      enable = true;
    };

    workspace = {
      onCreate = {
        install_python = "pip install -r requirements.txt";
        npm_install = "npm install";
      };
      onStart = {};
    };
  };
}
