# FASPR goes in this folder

Only needed for `epifitscout build --side-chains packed`. Everything else
in EpiFitScout works without it. FASPR is the recommended packer: one small C++
program, no registration, under a second per complex.

## 1. Build

FASPR is distributed as source, so you compile it once. Build it **outside this
repository** — the clone is also called `FASPR`, and cloning it here would
collide with this folder:

```bash
cd /tmp
git clone https://github.com/tommyhuangthu/FASPR.git
cd FASPR
g++ -O3 -o FASPR src/*.cpp
```

## 2. Install

Copy **both** files here. From the EpiFitScout repository root:

```bash
mkdir -p FASPR/bin
cp /tmp/FASPR/FASPR /tmp/FASPR/dun2010bbdep.bin FASPR/bin/
chmod +x FASPR/bin/FASPR
```

You should end up with:

```
FASPR/bin/FASPR
FASPR/bin/dun2010bbdep.bin     13 MB Dunbrack rotamer library
```

Both are required, and they must sit **in the same directory**. FASPR finds its
rotamer library at `dirname(argv[0])/dun2010bbdep.bin`, hard-coded, so a lone
binary — or a symlink pointing at one elsewhere — fails with
`error! cannot find rotamer library dun2010bbdep.bin`. Don't rename the library.

EpiFitScout also accepts `FASPR_BIN=/path/to/FASPR` or the binary on `PATH`; in
both cases the library still has to be beside the real executable.

FASPR is free for academic use (Huang, Pearce & Zhang, *Bioinformatics* 2020,
36:3758), which is why it is not redistributed here. Scwrl4 works too
(`--packer Scwrl4`, in `Scwrl4/bin/`) but requires registration.

## 3. Check

```bash
python check_setup.py
```

```
[ok]   FASPR      /…/epifitscout/FASPR/bin/FASPR
```

A missing or incomplete install is reported before any search starts, not
halfway through.

> The binary you build is for this machine's platform. Unlike the MASTER
> binaries, the source compiles on macOS and Windows as well.
