# FASPR goes in this folder

**Not needed to search.** EpiFitScout proper — stages 1 to 8,
`epifitscout.search()` and `epifitscout search` — ranks fragments on backbone
coordinates and never packs anything. You can skip this folder entirely if
retrieval is all you want.

**Needed to build models.** Stages 9 to 11 (`epifitscout.graft()`,
`epifitscout graft`, and everything in `example/`) rebuild side chains and
measure an all-atom interface. Without a packer the grafted loops are backbone
only, and every interface term then reads as a loose interface because the side
chains that decide complementarity are absent. Those runs still complete with
`--packer none`, but the numbers are diagnostic rather than results.

FASPR is the recommended packer: one small C++ program, no registration, under a
second per complex.

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

When FASPR is absent the check reports it as `[--]`, not `[FAIL]`, and still
says `Ready.` — a search does not need it:

```
[--]   FASPR      not installed — searching is unaffected; needed only to
                  build models (stages 9-11), which are backbone-only without it
```

Either way the state of the install is reported before any run starts, not
halfway through.

> The binary you build is for this machine's platform. Unlike the MASTER
> binaries, the source compiles on macOS and Windows as well.
