pwd := `pwd`
vcpkg_installed := pwd + '/vcpkg_installed/x64-linux'

setup build_dir='build/':
    meson setup \
        "-Dpkg_config_path={{vcpkg_installed}}/lib/pkgconfig/:{{vcpkg_installed}}/share/pkgconfig" \
        "-Dincludedir={{vcpkg_installed}}/include/" \
        "{{build_dir}}"

compile build_dir='build/':
    meson compile -C "{{build_dir}}"

vcpkg-install:
    vcpkg install
