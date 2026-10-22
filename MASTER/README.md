# MASTER binaries go in this folder

EpiFitScout drives MASTER (Zhou & Grigoryan, *Protein Science* 2015) to do its
structural search. MASTER is a separate program and is not part of this
repository, so you add it once.

## 1. Download

<https://grigoryanlab.org/legacy.php?sec=download&soft=MASTER>

Name, institution and email are optional; answer the small arithmetic question.
Choose the **binaries**, not the source:

> **`master-bin-v1.6.tar.gz`**

(MASTER v1.5 also works. This is MASTER **v1**; v2 is a different program and is
not supported.) MASTER v1 is distributed under the GPL for academic use; for
commercial use, contact the authors.

## 2. Unpack

From the repository root:

```bash
tar -xzf master-bin-v1.6.tar.gz -C MASTER/
chmod +x MASTER/bin/*
```

You should end up with:

```
MASTER/bin/master
MASTER/bin/createPDS
MASTER/bin/parsePDS        (not used by EpiFitScout)
```

## 3. Check

```bash
python check_setup.py
```

> The binaries are Linux x86-64 executables. They will not run on macOS or
> Windows.
