# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

SRC_DIR = os.path.abspath('.')

a = Analysis(
    ['server.py'],
    pathex=[SRC_DIR],
    binaries=[],
    datas=[
        (os.path.join(SRC_DIR, 'index.html'), '.'),
        (os.path.join(SRC_DIR, 'src'), 'src'),
        (os.path.join(SRC_DIR, 'models'), 'models'),
        (os.path.join(SRC_DIR, 'test_examples'), 'test_examples'),
        (os.path.join(SRC_DIR, 'train_examples'), 'train_examples'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='emotion_classifier',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='favor.ico'
)
