%% -*- coding: utf-8; mode: latex -*- %%
%
% Created by David M. M. de Matos (david.matos@acm.org).
% Copyright 1993-2010 by David Martins de Matos.
% All rights reserved.
%
% This is a LaTeX2e class file.
% It will NOT work with plain TeX or LaTeX 2.09.
%
% The thesis is formated as a book, with Portuguese labels.
% ... or in English
%
% Known bugs: comments should be in Portuguese (or in English...)
%
% Old version: 1994/06/15 - 2003/10/29 15:57
% Previous version: 2003/11/10 18:08
% Previous version: 2005/03/20 10:21
% Previous version: 2007/05/24 15:39
% Previous version: 2010/09/14 20:05
% This version: 2012/02/27 13:25
%

\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{thesis}[2020/01/08 14:46
 David's excellent thesis template!]

%---------------------------------------------------------------------------
% A thesis is a book...
% I want to include Encapsulated PostScript pictures in the document,
% and use Portuguese labels. Portuguese hyphenation will be automatically
% activated if the format contains the proper hyphenation patterns.

% The doublespace package allows for flexible inter-line distance control.
\RequirePackage{doublespace}          % Allow line-spacing control...

%---------------------------------------------------------------------------
% Options...

% old options: they are now passed to the base class
%\DeclareOption{draft}{}
%\DeclareOption{final}{}

% Never indent...
\DeclareOption{indent}{%
  \let\@afterindentfalse\@afterindenttrue%
  \@afterindenttrue%
}

% Line spacing...
\DeclareOption{singlespace}{\setstretch{1.0}}
\DeclareOption{doublespace}{\setstretch{1.5}}

\DeclareOption*{%
    \PassOptionsToClass{\CurrentOption}{book}%
}

\ExecuteOptions{singlespace}%
\ProcessOptions\relax

\LoadClass{book}

%---------------------------------------------------------------------------

% Encoding.
\usepackage[utf8]{inputenc}
\usepackage{array}
\newcolumntype{L}{>{\centering\arraybackslash}m{11cm}}

% Fonts...
% Palatino is used to write everything. Well, almost. Courier will be used
% as typewriter font (fixed width) and Helvetica as sans serif font.
\usepackage{palatino}
%\usepackage{times}
%\usepackage{utopia}

% Use color...
%\usepackage[dvips]{color}
\usepackage[pdftex]{color}

% the following lines are a horrible hack!
% they are here to allow us to use color in titles and
% to allow the use of figures.
% the bad side of these lines is that probably color boxes
% won't work...
\def\color@setgroup{}
\def\color@begingroup{}
\def\color@endgroup{}
\def\color@hbox{}
\def\color@vbox{}
\def\color@endbox{}

%\usepackage{psboxit}              % Allow PS boxes...
\usepackage{multicol}             % Allow multicolumn output...

\usepackage[noprefix]{nomencl}    % glossary/nomenclature
\usepackage[acronym,section=section]{glossaries}

%---------------------------------------------------------------------------
% Acronyms package - GLS

\loadglsentries{acronyms/acronyms}
\makeglossaries

% Bibliography styles...

%--------
% Bib: use authordate
%\usepackage{authordate} % Author/Date bibliography
%%\def\indexname{Índice Remissivo}
%\usepackage{index}                              % don't use with apacite
%% \def\indexname{Índice Remissivo}
%\newindex{default}{idx}{ind}{Índice Remissivo}  % don't use with apacite

%--------
% Bib: use chicago
%\usepackage{chicago} % Chicago Manual of Style
%%\def\indexname{Índice Remissivo}
%\usepackage{index}                              % don't use with apacite
%% \def\indexname{Índice Remissivo}
%\newindex{default}{idx}{ind}{Índice Remissivo}  % don't use with apacite

%--------
\usepackage[colorlinks,linkcolor=blue,hyperindex,plainpages=false,pdfpagelabels]{hyperref}
% Bib: use apacite
\usepackage[stdindex,hyper]{apacite}
\renewcommand{\bibname}{Author Index}
\renewcommand{\BBAB}{e} % between authors in text (apacite)
\makeindex
\def\Hy@reserved@a{}                                    % USE with apacite

% Other stuff
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{wrapfig}
\usepackage{url}
\usepackage{graphicx}
\usepackage{csvsimple}
%\usepackage[hyperindex]{hyperref}

%---------------------------------------------------------------------------

%\def\contentsname{Índice}

%---------------------------------------------------------------------------
% The following code replaces the corresponding code in bk??.clo
% This code is liable to be depend heavily on the *.clo files...

%\font\ch@pnumfont=putbo8r at 2in %cminch
%\font\ch@pnumfont=pplbo at 2in %cminch
%\font\ch@pnumfont=eurb9 at 2in %cminch
\font\ch@pnumfont=eurm10 at 2in %cminch -> ams euler

\newbox\s@ch@@ghtb@x
\setbox\s@ch@@ghtb@x=\hbox{{\normalfont\Huge A}}

%FEIA->\font\ch@pnamfont=pplb at 8mm % palatino bold
\font\ch@pnamfont=cmbr8 at 8mm % computer modern bright
\font\p@rtnamfont=cmbr8 at 13mm % computer modern bright
\font\s@cnamfont=cmbrsl8 at \ht\s@ch@@ghtb@x % computer modern bright

% Color...
\definecolor{chapheadcolor}{rgb}{.7,.8,.9}

\def\@makechapterhead#1{
{
  %\addcontentsline{lof}{chapter}{#1}%
  %\addcontentsline{lot}{chapter}{#1}%
        \thispagestyle{empty}\hfill
        \vbox to1.5in{
                \vfil
                \hbox to0.8\textwidth{
                        \hfil
                        \color{chapheadcolor}%
                        {\ch@pnumfont\thechapter}
                        \hbox to0.05\textwidth{}
                }
                \vfil
        }
        \vskip-1.5in
        \vbox to1.5in{
                \vfil
                \hbox to\textwidth{
                        \hfil
                        \ch@pnamfont
                        \begin{minipage}{0.6\textwidth}
                                \begin{flushright}
                                        \baselineskip=30pt
                                        #1
                                \end{flushright}
                        \end{minipage}
                }
                \vskip18pt
                %\vfil
        }
}%
%\pagebreak[4]%
}

%---------------------------------------------------------------------------

\newbox\@nv@s@bl@
\setbox\@nv@s@bl@=\hbox{\ch@pnumfont A}

\def\@makeschapterhead#1{
{
        \hfill
        \vbox to\ht\@nv@s@bl@{}
        \vskip-1.5in
        \vbox to1.5in{
                \vfil
                \hbox to\textwidth{
                        \hfil
                        \ch@pnamfont
                        \begin{minipage}{0.8\textwidth}
                                \begin{flushright}
                                        \baselineskip=30pt
                                        #1
                                \end{flushright}
                        \end{minipage}
                }
                \vskip18pt
                %\vfil
        }
}}

%---------------------------------------------------------------------------

\def\@part[#1]#2{%
\thispagestyle{empty}
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \markboth{}{}%
    {\centering
     \interlinepenalty \@M
     \normalfont
     \ifnum \c@secnumdepth >-2\relax
        {%
       \color{chapheadcolor}{\ch@pnumfont \thepart}}
       \par
       \vskip -50\p@
     \fi
     {\p@rtnamfont #2}\par}%
    \@endpart}

%---------------------------------------------------------------------------

\newdimen\thesis@marginglue
\thesis@marginglue=0.3in

\newdimen\s@cskip
\s@cskip=1.6\thesis@marginglue

%\renewcommand\section{\@startsection {section}{1}{-\s@cskip}%
\renewcommand\section{\@startsection {section}{1}{0pt}%
                                   {-3.5ex plus -0.5ex minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\s@cnamfont}}
\renewcommand\subsection{\@startsection{subsection}{2}{0pt}%
                                     {-2.5ex plus -0.5ex minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\large\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{0pt}%
                                     {-2.5ex plus -0.5ex minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{0pt}%
                                    {0.25ex plus 0.5ex minus.2ex}%
                                    {-1em}%
                                    {\normalfont\normalsize\bfseries}}
\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {0.25ex plus 0.5ex minus .2ex}%
                                       {-1em}%
                                      {\normalfont\normalsize\bfseries}}

%---------------------------------------------------------------------------

\def\appendixname{Appendix}
\renewcommand\appendix{\par
  \setcounter{chapter}{0}
  \setcounter{section}{0}
  \def\@chapapp{\appendixname}
  \def\thechapter{\Alph{chapter}}}

%---------------------------------------------------------------------------

\newenvironment{theauthorindex}
               {\def\indexname{Author Index}
                \chapter*{\indexname}
                \@mkboth{\MakeUppercase\indexname}{\MakeUppercase\indexname}
                \parskip\z@ \@plus .3\p@\relax
                \let\item\@idxitem\begin{multicols}{3}\raggedright}{\end{multicols}}

\renewenvironment{theindex}
               {%\def\indexname{Índice Remissivo}
                \chapter*{\indexname}
                \@mkboth{\MakeUppercase\indexname}{\MakeUppercase\indexname}
                \parskip\z@ \@plus .3\p@\relax
                \let\item\@idxitem\begin{multicols}{3}\raggedright}{\end{multicols}}

\renewenvironment{thenomenclature}
               {\def\glossaryname{Nomenclature}
                \chapter{\glossaryname}
                \@mkboth{\MakeUppercase\glossaryname}{\MakeUppercase\glossaryname}
                {Este capítulo apresenta alguma da terminologia utilizada na dissertação. Alguns dos termos apresentados uesultam da tradução de termos utilizados na literatura internacional.}
                \begin{description}\sloppy}{\end{description}}
%                \begin{multicols}{2}\begin{description}\sloppy}{\end{description}\end{multicols}}

%---------------------------------------------------------------------------
% Bibliography styles...

\newcommand{\bibtitle}{Bibliography}
\newcommand{\bibheadtitle}{BIBLIOGRAPHY}

% Redefine the bibliography environment to generate a new chapter,
% with the corresponding index entry...

\renewenvironment{thebibliography}{
  \chapter*{\bibtitle}%
  \@mkboth{\bibheadtitle}{\bibheadtitle}%
  \list{\relax}{\setlength{\labelsep}{0em}
    %--apacite \setlength{\itemindent}{-\bibhang}
    %--apacite \setlength{\leftmargin}{\bibhang}%
    \setlength{\itemindent}{1.3\bibindent}
  }
  %\def\newblock{\hskip .11em plus .33em minus .07em}
  %\sloppy\clubpenalty4000\widowpenalty4000
  \sfcode`\.=1000
  }{}%1col.
  %--apacite \columnsep=1.3\bibhang}{}%1col.
  %\columnsep=1.3\bibhang\begin{multicols}{2}}{\end{multicols}}%2col.

%---------------------------------------------------------------------------
% More settings...
\newcommand{\pdash}{\discretionary{-}{-}{-}}

%---------------------------------------------------------------------------
% More settings...
\pagestyle{empty}

%---------------------------------------------------------------------------
% Vertical distances...

\headsep=12mm          % distance between header and text
\topmargin=-5mm
\textheight=23cm
\textwidth=16cm

%\textheight=18.5cm
%\textwidth=11.5cm

\oddsidemargin=0pt
\evensidemargin=0pt

%---------------------------------------------------------------------------
% Horizontal distances...

\parskip=7pt plus 2pt minus 1pt
\parindent=7mm

\setcounter{tocdepth}{4}
\setcounter{secnumdepth}{4}
%\pretolerance=1000
\makenomenclature
\sloppy

%---------------------------------------------------------------------------
% Extra...
% This provides a bit of useless information.

%\listfiles

%%---------------------------------------------------------------------------
