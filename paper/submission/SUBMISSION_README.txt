IEEE ACCESS SUBMISSION PACKAGE
==============================

Manuscript : Evaluation Artefacts in Physics-Informed Band-Gap Prediction:
             A Leakage-Controlled Study of Two-Dimensional Materials
Authors    : Adhvay Jagadeesh, Rutvi Mudalagi, Marx Akl (corresponding)
Template   : IEEE Access (ieeeaccess.cls)

This directory is self-contained. It has been verified to compile from a
fresh copy with no cached artefacts, and to resolve every citation from the
bundled main.bbl even when BibTeX is not run -- which is how ScholarOne's
LaTeX compiler sometimes behaves.

BUILD
-----
    pdflatex main
    bibtex   main        (optional; main.bbl is already present)
    pdflatex main
    pdflatex main

Expected output: main.pdf, 11 pages, 10 figures, 5 tables, 12 equations.

CONTENTS
--------
    main.tex                 manuscript source
    main.bbl                 pre-built bibliography (53 entries, IEEEtran style)
    references.bib           BibTeX source for the above
    main.pdf                 compiled reference copy
    graphical_abstract.png   graphical abstract, 3600 x 2025 px (upload separately)
    graphical_abstract.pdf   same, vector
    graphical_abstract_caption.txt
    figures/fig1 ... fig10   all figures, PNG, 300 dpi
    ieeeaccess.cls           IEEE Access class (unmodified, from the template)
    IEEEtran.cls, .bst       dependencies of the class
    spotcolor.sty            dependency of the class
    logo.png, bullet.png,
    notaglinelogo.png        template assets
    t1-*.pfb/.tfm/.map/.fd   template fonts (Formata, Times, Giovanni)

UPLOAD TO SCHOLARONE AS
-----------------------
    Main Document          main.pdf
    Source files           this whole directory, zipped
    Figure files           figures/*.png   (ScholarOne may ask for each
                           figure separately; they are already at 300 dpi)

STILL REQUIRED, NOT IN THIS PACKAGE
------------------------------------
These are entered in the ScholarOne forms or uploaded separately; they are
not part of the LaTeX source.

  [x] Graphical abstract  -- graphical_abstract.png (3600 x 2025 px, 300 dpi)
                             and graphical_abstract.pdf (vector) in this
                             directory. Caption in graphical_abstract_caption.txt
                             (98 words). Regenerate with
                             experiments/make_graphical_abstract.py; every
                             number is read from the result artefacts.
  [ ] ORCID for every author (linked in each author's ScholarOne profile;
                             the submission cannot proceed without them)
  [ ] Cover letter        -- draft exists in the project as
                             Cover_Letter_IEEE_Access.docx
  [ ] Suggested reviewers -- three to five, no conflicts of interest
  [ ] Funding statement   -- currently absent; state "none" if applicable
  [ ] Author biographies  -- present in main.tex but brief (one to two
                             sentences each). IEEE Access expects three to
                             four sentences with affiliation history and
                             research interests. Expand before submission.
  [ ] Data availability   -- repository URL is in the manuscript:
                             https://github.com/adhvayjagadeesh/DFT---Machine-Learning--PINN
                             Consider archiving a release on Zenodo for a
                             citable DOI.

PLACEHOLDERS LEFT DELIBERATELY IN main.tex
------------------------------------------
    \doi{10.1109/ACCESS.2026.0000000}   -- assigned by IEEE after acceptance
    The \history line has been removed at the authors' request; IEEE
    inserts publication dates at production.
