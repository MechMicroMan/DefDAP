defaults = {
    # Convention to use when attaching an orthonormal frame to a crystal
    # structure. 'hkl' or 'tsl'
    # OI/HKL convention - x // [10-10],     y // a2 [-12-10]
    # TSL    convention - x // a1 [2-1-10], y // [01-10]
    'crystal_ortho_conv': 'hkl',
    # Projection to use when plotting pole figures. 'stereographic' (equal
    # angle), 'lambert' (equal area) or arbitrary projection function
    'pole_projection': 'stereographic',
}
