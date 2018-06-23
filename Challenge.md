# Puzzles Challenge

Imagine taking a photo of a puzzle, thinking really hard for a couple
seconds, and then knowing exactly what picture the puzzle will show
when completed.

Now stop imagining it and make a computer program that does this!

Since this challenge is not intended to take up a huge amount of your
time, we've decided to have you build one smallish component of such a
system: given two square puzzle pieces from the same puzzle, determine
whether they match each other or not.

If you find yourself with time left over, feel free to get started on
the puzzle solving task, but we definitely don't expect you to solve
it in the allotted time.

## The Data

* There are twenty puzzles (shown in both their assembled and
  disassembled state) in the `data/train` folder.

* There are another twenty puzzles shown only in their disassembled
  state, in the `data/validate` folder.

* The puzzles in their disassembled state are a collection of square
  20x20 pixel tiles, but some of them are missing. Note that the tiles
  are in their original orientation, rather than rotated.

* You may use any auxilliary datasets you choose if your approach
  requires such a thing.

* Your algorithm will be evaluated on a holdout set by measuring the
  accuracy of your match vs. mismatch classifier. Don't worry about
  getting the best possible score though - if your approach is
  promising and is returning semi-reasonable results, that's good too.

## The Deliverables

* Classification of puzzle piece pairs as either matching
  horizontally, vertically, or not at all.

* The code used to generate the above classifications.

* A makefile that will build the project, including installing any
  dependencies.

* A simple readme file describing the use of the project, including
  any training that should be run.

* A list of any external datasets that you used.

* Any documentation you want to provide that describes the choices
  made.

## Keep in Mind

We're looking for more modern / innovative approaches. Please make
sure to point out and cite any novel ideas that were incorporated in
your answer.

Feel free to make use of any libraries you prefer (Keras & TF,
pyTorch, Caffe/Caffe2), and any pre-trained weights you think will be
useful.

If you've previously built something similar, feel free to adapt
(though remember that we're expecting a Python package, and please
follow best coding & documentation practices : )

Unit tests are optional, but appreciated.
