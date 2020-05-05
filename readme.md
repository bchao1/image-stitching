###To run:

```bash
$ ./run.sh
```

`run.sh`：

```bash
python3 main.py 4 5000 lib1 --use_cache
```

main.py usage:

`usage: python3 main.py [-h] [--use_cache] ratio f run`

-  args:
  1. ratio : Downscaling ratio. 
  2. f : Focal length, in pixels.
  3. run : Testing set to run.
  4. --use_cache : Use pre-computed features. Advised.

###Original pictures :

located in `runs / lib1 / images`

### Final result:

`./final.jpg`

