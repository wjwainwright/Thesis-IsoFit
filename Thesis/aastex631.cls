%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  AASTEX631.cls                                 %%
%%  Jan 29f, 2021                                  %% 
%%                                                %%
%% Copyright 2021 American Astronomical Society   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\def\currversion{6.3.1f}

%%
%%    These files are distributed
%%    WITHOUT ANY WARRANTY; without even the implied warranty of
%%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%% 
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}%
%%
%%%     date            = "9/19/2016",
%%%     author          = "Amy Hendrickson (mailto:amyh@texnology.com)" 
%%%                            including 
%%%                       1) code from emulateapj.cls, version 01/23/15(marked below)
%%%                            written by Alexey Vikhlinin, which relies on RevTeX4-1
%%%                            for much functionality;
%%%                       2) some code from aastex (marked below)
%%%                       3) Original work by Amy Hendrickson (marked below) for
%%%                       extensive table tools, including easy decimal entry and splitting
%%%                       tables horizontally into 2 or 3 pieces, for tabular and deluxetable
%%%                       environments; rotate tables; gridlines for use in positioning figures, 
%%%                       track changes.

%%% This work may be distributed and/or modified under the 
%%% conditions of the LaTeX Project Public License, either version 1.3c 
%%% of this license or (at your option) any later version.
%%% The latest version of this license is in
%%%  http://www.latex-project.org/lppl.txt
%%% and version 1.3 or later is part of all distributions of LaTeX 
%%% version 2005/12/01 or later.
%
%%% This work has the LPPL maintenance status `maintained'.
%
%%% The Current Maintainer of this work is Greg Schwarz <greg.schwarz@aas.org> 
%%% This work consists of the file aastex.cls

%%%     copyright       = "Copyright (C) 2016--2020 American Astronomical Society
%%%
%%%                        This work may be distributed and/or modified under the
%%%                        conditions of the LaTeX Project Public License, either version 1.3
%%%                        of this license or (at your option) any later version.
%%%                        The latest version of this license is in
%%%                        http://www.latex-project.org/lppl.txt
%%%                        and version 1.3 or later is part of all distributions of LaTeX
%%%                        version 2003/12/01 or later.
%%%
%%%                        This work has the LPPL maintenance status "maintained".
%%%
%%%                        The Current Maintainer of this work is the American Astronomical Society.
%%%
%%%                        This work consists of all files listed in the document README.
%%%
%%%     address         = "American Astronomical Society
%%%                        USA",
%%%     telephone       = "+1 ???",
%%%     FAX             = "",
%%%     email           = "aastex-help@aas.org",
%%%     codetable       = "ISO/ASCII",
%%%     keywords        = "latex, AAS, journal",
%%%     supported       = "yes",
%%%     abstract        = "formatter for AAS journal submissions",
%%%  }


\ProvidesClass{aastex631}%%%
 [2020/12/20 Version 6.3.1d/AAS markup document class]%
{}
\ClassInfo{aastex}{%
^^J
^^J
 Original \protect\LaTeX2.09 style
 by Chris Biemesderfer (chris@seagoat.com).
^^J
 Adapted to \protect\LaTeXe\space
 by A. Ogawa (ogawa@teleport.com)%
^^J
emulateapj.cls included, Copyright 2000-2015 Alexey Vikhlinin
^^J
Rewrite and update of emulateapj.cls,
revised and enhanced with table macros, gridlines, draft watermark,^^J
track changes, new `modern' style and much more, by Amy Hendrickson,%
^^J
 (amyh@texnology.com, http://www.texnology.com)
^^J
^^J
}%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% aastex.cls version 2.0 is a based on emulateapj.cls, version January 23, 2015,
%% and an older version of aastex.cls, with many additional functionalities.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Options. Document style options, and additional feature options.

% These are the General document style options, 6 choices

%   manuscript	: onecolumn, doublespace, 12pt fonts

%%  Preprints are like manuscript, except they are single spaced.
%   preprint	: onecolumn, single space, 12pt fonts
%   preprint2	: twocolumn, single space, 12pt fonts

%% Modern is new design developed by David Hogg and Daniel Foreman-Mackey
%   modern      : one column, single space, 12pt fonts, more stylish

%   twocolumn	: a two column article, single space, 10pt  fonts

% Default document style option
%   onecolumn	: a one column article; single space, 10pt fonts

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Additional feature options:

%   revtex4 	: will produce an error message and demand to download revtex4-1. 
%   twocolappendix: make 2 column appendix // produces error message. 
%      onecolappendix: make 1 column appendix is the default. // not needed because it is the default
%      astrosymb	: Loads Astrosymb font and define \astro commands. 
%      tighten	: Makes baselineskip slightly smaller, only works with twocolumn option.
%%    author can also use the commands \tighten, or \singlespace or \doublespace in the body of their article
%%    to get this change to the baselineskip
%      times	: uses times font as the default
%      linenumbers	: turn on lineno package.
%      trackchanges : required to see the revision mark up and print output
%      letteredappendix: (was numbered appendix) Not needed because it is the default, will label appendix sections A, B, ...
%      numberedappendix: Legacy command, will produce same results as lettered appendix, since we don't want more
%                        than one `section 1' in article: will cause problems for cross referencing.

%    anonymous: to not list authors/affiliations be listed

\newif\ifmanu
\newif\iftwelvepoint
\newif\ifmodern
\newif\ifdoublespace
\newif\if@two@col
\newif\ifonecolstyle
\newif\iftwocolstyle

\newif\ifrnaas
%% makes modern the style for rnaas
\DeclareOption{RNAAS}{\global\rnaastrue}
\DeclareOption{rnaas}{\global\rnaastrue}

 \DeclareOption{manuscript}{\twelvepointtrue\@two@colfalse\doublespacetrue\manutrue}%

%% Preprints are like manuscript but are single spaced:
\newif\ifpreprint
 \DeclareOption{preprint}{\@two@colfalse\preprinttrue\twelvepointtrue}%

\newif\ifpreprinttwo
 \DeclareOption{preprint2}{\@two@coltrue\preprinttwotrue\twelvepointtrue}% 


%% New design suggested by
\DeclareOption{modern}{\@two@colfalse\twelvepointtrue\moderntrue}

%% left in from older version, in case it could be useful:



\DeclareOption{twocolumn}{\onecolstylefalse\twocolstyletrue\@two@coltrue\twelvepointfalse}

%% default style
\DeclareOption{onecolumn}{\onecolstyletrue\twocolstylefalse}

%% obsolete option
\DeclareOption{revtex4}{\typeout{=================^^J^^J!!! The revtex4 option no longer
 allowed!^^J^^J Please download revtex4-1.cls^^J^^J
 Thank-you!^^J^^J=================}\stop}

\newif\if@two@col@app
\DeclareOption{twocolappendix}{\@two@col@apptrue}
\DeclareOption{onecolappendix}{\@two@col@appfalse}%% this is default

\newif\iflongauthor
\DeclareOption{longauthor}{\global\longauthortrue}

\newlength{\bibbaselineskip}\setlength{\bibbaselineskip}{3.075mm}

 \DeclareOption{astrosymb}{
   \def\astro#1{\leavevmode\hbox{\astro@font#1}}%
   \def\load@astro@font{%
\iftwelvepoint
 \font\astro@font=astrosym at 8pt
\else
 \font\astro@font=astrosym at 7pt
\fi
}%
   \AtBeginDocument{\load@astro@font}
 }

%% Default definition for \astro, in case astrosymb option has not
%% been used:
\def\astro#1{\typeout{^^J^^J Please use class option `astrosymb' to
access fonts for the \string\astro\space command
^^J^^J
}\stop}

   \def\apjsecfont{\normalsize}
   \def\secnum@size{\small}
   \def\subsecnum@size{\normalsize}

    \def\AppendixApjSectionMarkInTitle#1{\relax}
\newif\ifletteredappendix
     \DeclareOption{letteredappendix}{\global\letteredappendixtrue
       \def\AppendixApjSectionMarkInTitle#1{#1.\ }
     }

%% this option is obsolete, since letteredappendix is a more logical name;
%%  but keeping it here in case someone uses it, set to be the same as
%% letteredappendix.
     \DeclareOption{numberedappendix}{\global\letteredappendixtrue
       \def\AppendixApjSectionMarkInTitle#1{#1.\ }
     }

     \newif\if@number@appendix@floats\@number@appendix@floatsfalse
     \DeclareOption{appendixfloats}{
       \@number@appendix@floatstrue
     }

%%%%%%%%%%%%%
%% AH

%% This option changes the baselineskips in the definitions of \normalsize etc
%% to make them a bit smaller.
\newif\iftighten
\DeclareOption{tighten}{\global\tightentrue}

%% AH
\newif\iftimes
\DeclareOption{times}{\global\timestrue}

\newif\ifnumlines
\DeclareOption{linenumbers}{\global\numlinestrue}

\newif\iftrack
\DeclareOption{trackchanges}{\global\tracktrue}

\newif\ifanonymous
\DeclareOption{anonymous}{\global\anonymoustrue}


%% End AH Options

\ExecuteOptions{onecolumn,onecolappendix,letteredappendix}% 

\ProcessOptions	

%%%%%%%%%%%%%
%

 \iftimes
%% Might be better to use apjfonts, from Alexey Vikhlinin's website
%% http://hea-www.harvard.edu/~alexey/emulateapj/apjfonts.sty
%%\usepackage{apjfonts}
%% but times.sty is ubiquitous, and many people may not have apjfonts.sty
%% or the fonts that it calls.
%% default fonts:
\usepackage{times}
 \fi

%%%%%%%%%%%%%

 \IfFileExists{revtex4-1.cls}{
    \def\@revtex@cls{revtex4-1}
  }{
\typeout{^^J^^J Please update your system to include
revtex4-1.cls^^J^^J}\stop
  }
\def\@revtex@cls{revtex4-1}
\let\@startsection@from@latex=\@startsection


%% Passing options to revtex while loading:
  \iftwelvepoint
      \iflongauthor
         \LoadClass[nofootinbib,showkeys,twoside,floatfix,unsortedaddress,flushbottom,12pt,aps,pra]
          {\@revtex@cls}
     \else
        \LoadClass[nofootinbib,showkeys,twoside,floatfix,superscriptaddress,flushbottom,12pt,aps,pra]
       {\@revtex@cls}
     \fi
  \else %not twelvepoint
     \iflongauthor
         \LoadClass[nofootinbib,showkeys,twoside,floatfix,unsortedaddress,flushbottom,10pt,aps,pra]
         {\@revtex@cls}
     \else
        \LoadClass[nofootinbib,showkeys,twoside,floatfix,superscriptaddress,flushbottom,10pt,aps,pra]
        {\@revtex@cls}
    \fi
  \fi

%%%%%%%%%%%%%%%%%%%%

\def\blankaffiliation{~}%

%% needed?
 \def\@eapj@cap@font{\bfseries}
 \def\@eapj@figname{Figure}
 \def\@eapj@tabname{Table}

% Restore natbib package without sorting. 
\def\NAT@sort{0}
\def\NAT@cmprs{0}

% only redefine sort@cites for natbib versions < 2009
\@ifundefined{NAT@sort@cites@}{\def\NAT@sort@cites#1{\edef\NAT@cite@list{#1}}}%

\RequirePackage{latexsym}% 
\RequirePackage{graphicx}% 
\RequirePackage{amssymb}% 

\newcommand{\bibstyle@aas}{\bibpunct{(}{)}{;}{a}{}{,}}% 
\@citestyle{aas}% 

\let\@startsection=\@startsection@from@latex
\def\baselinestretch{1.0}

%% Default vertical space between horizontal lines. Can change definition
%% of arraystretch in the body of the paper; either for full article or
%% for a single table.
\AtBeginDocument{\def\arraystretch{1}}

%% needed?
\def\raggedcolumn@sw#1#2{#2}  % implement flushbottom as it was in revtex4

\RequirePackage{epsf,graphicx}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Font families
%% modification of bk10.clo, and bk12.clo, standard LaTeX distribution
%%
%% Implements doublespace used in `manuscript' option
%% Implements `tighten' option: make baselineskip smaller if `tighten' is used
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% tighten only works with two column styles:
\iftwocolstyle\else\global\tightenfalse\fi

%%% preprint, preprint2, manuscript are 12pt
%% Modified bk12.clo
\iftwelvepoint

\renewcommand\normalsize{%
\ifdoublespace % manuscript
  \iftighten
   \@setfontsize\normalsize\@xiipt{20}%
  \else
   \@setfontsize\normalsize\@xiipt{24}%
  \fi
\else
  \iftighten
   \@setfontsize\normalsize\@xiipt{13}%
   \else
      \ifmodern
      \@setfontsize\normalsize\@xiipt{16}%
      \else
        \@setfontsize\normalsize\@xiipt{15}%
      \fi
   \fi
\fi %% end doublespace/no double space
   \abovedisplayskip 12\p@ \@plus3\p@ \@minus7\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@
   \belowdisplayskip \abovedisplayskip
   \let\@listi\@listI}

\normalsize

\renewcommand\small{%
\ifdoublespace % manuscript
   \@setfontsize\small\@xipt{20}%
\else
\ifmodern
   \@setfontsize\small\@xipt{12}%
\else
   \@setfontsize\small\@xipt{13.6}%
\fi\fi
%% end doublespace/no double space
   \abovedisplayskip 11\p@ \@plus3\p@ \@minus6\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 9\p@ \@plus3\p@ \@minus5\p@
               \parsep 4.5\p@ \@plus2\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}

\renewcommand\footnotesize{%
\ifdoublespace % manuscriptstyle
   \@setfontsize\footnotesize\@xpt{18}
\else
   \ifmodern
   \@setfontsize\footnotesize\@xpt{11}%
   \else
   \@setfontsize\footnotesize\@xpt\@xiipt
\fi\fi
   \abovedisplayskip 10\p@ \@plus2\p@ \@minus5\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 6\p@ \@plus2\p@ \@minus2\p@
               \parsep 3\p@ \@plus2\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}
\ifdoublespace % manuscript style
\renewcommand\scriptsize{\@setfontsize\scriptsize\@viiipt{14}}
\renewcommand\tiny{\@setfontsize\tiny\@vipt{10}}
\renewcommand\large{\@setfontsize\large\@xivpt{18}}
\renewcommand\Large{\@setfontsize\Large\@xviipt{22}}
\renewcommand\LARGE{\@setfontsize\LARGE\@xxpt{25}}
\renewcommand\huge{\@setfontsize\huge\@xxvpt{30}}
\else
\renewcommand\scriptsize{\@setfontsize\scriptsize\@viiipt{9.5}}
\renewcommand\tiny{\@setfontsize\tiny\@vipt\@viipt}
\renewcommand\large{\@setfontsize\large\@xivpt{22}}
\renewcommand\Large{\@setfontsize\Large\@xviipt{24}}
\renewcommand\LARGE{\@setfontsize\LARGE\@xxpt{28}}
\renewcommand\huge{\@setfontsize\huge\@xxvpt{30}}
\fi
\let\Huge=\huge
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\else %% not twelvepoint, now tenpoint
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% no doublespace here, because the only doublespace is in `manuscript style' which is 12pt

\renewcommand\normalsize{%
\iftighten
   \@setfontsize\normalsize\@xpt{11.5}%
\else
   \@setfontsize\normalsize\@xpt{12.5}%
\fi
   \abovedisplayskip 10\p@ \@plus2\p@ \@minus5\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@
   \belowdisplayskip \abovedisplayskip
   \let\@listi\@listI}

\normalsize

\renewcommand\small{%
\iftighten
   \@setfontsize\small\@ixpt{12}%
\else
   \@setfontsize\small\@ixpt{11}%
\fi
   \abovedisplayskip 8.5\p@ \@plus3\p@ \@minus4\p@
   \abovedisplayshortskip \z@ \@plus2\p@
   \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 4\p@ \@plus2\p@ \@minus2\p@
               \parsep 2\p@ \@plus\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}

\renewcommand\footnotesize{%
\iftighten
   \@setfontsize\footnotesize\@viiipt{9}%
\else
   \@setfontsize\footnotesize\@viiipt{9.5}% standard eightpoint size
\fi
   \abovedisplayskip 6\p@ \@plus2\p@ \@minus4\p@
   \abovedisplayshortskip \z@ \@plus\p@
   \belowdisplayshortskip 3\p@ \@plus\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 3\p@ \@plus\p@ \@minus\p@
               \parsep 2\p@ \@plus\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}

%% Standard 10pt font sizes
\renewcommand\scriptsize{\@setfontsize\scriptsize\@viipt\@viiipt}
\renewcommand\tiny{\@setfontsize\tiny\@vpt\@vipt}
\renewcommand\large{\@setfontsize\large\@xiipt{14}}
\renewcommand\Large{\@setfontsize\Large\@xivpt{18}}
\renewcommand\LARGE{\@setfontsize\LARGE\@xviipt{22}}
\renewcommand\huge{\@setfontsize\huge\@xxpt{25}}
\renewcommand\Huge{\@setfontsize\Huge\@xxvpt{30}}

\fi

%% End font sizes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\setlength\smallskipamount{3\p@ \@plus 1\p@ \@minus 1\p@}
\setlength\medskipamount{6\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\bigskipamount{12\p@ \@plus 4\p@ \@minus 4\p@}
\setlength\headheight{12\p@}
\def\@listi{\leftmargin\leftmargini
            \parsep 4\p@ \@plus2\p@ \@minus\p@
            \topsep 8\p@ \@plus2\p@ \@minus4\p@
            \itemsep4\p@ \@plus2\p@ \@minus\p@}
\let\@listI\@listi
\@listi
\def\@listii {\leftmargin\leftmarginii
              \labelwidth\leftmarginii
              \advance\labelwidth-\labelsep
              \topsep    4\p@ \@plus2\p@ \@minus\p@
              \parsep    2\p@ \@plus\p@  \@minus\p@
              \itemsep   \parsep}
\def\@listiii{\leftmargin\leftmarginiii
              \labelwidth\leftmarginiii
              \advance\labelwidth-\labelsep
              \topsep    2\p@ \@plus\p@\@minus\p@
              \parsep    \z@
              \partopsep \p@ \@plus\z@ \@minus\p@
              \itemsep   \topsep}
\def\@listiv {\leftmargin\leftmarginiv
              \labelwidth\leftmarginiv
              \advance\labelwidth-\labelsep}
\def\@listv  {\leftmargin\leftmarginv
              \labelwidth\leftmarginv
              \advance\labelwidth-\labelsep}
\def\@listvi {\leftmargin\leftmarginvi
              \labelwidth\leftmarginvi
              \advance\labelwidth-\labelsep}

%  ****************************************************
%  *             TWO AND SINGLE ONE COLUMN STYLES: AV *
%  ****************************************************
%

%%%%% include atbeginend.sty by AV:

% Copy \begin and \end to \BeginEnvironment and \EndEnvironment
\let\BeginEnvironment=\begin
\let\EndEnvironment=\end
%% \ifundefined from TeXbook
\def\IfUnDef#1{\expandafter\ifx\csname#1\endcsname\relax}
% Null command needed to eat its argument
\def\NullCom#1{}
\def\begin#1{%
% if defined \BeforeBeg for this environment, execute it
\IfUnDef{BeforeBeg#1}\else\csname BeforeBeg#1\endcsname\fi%
\IfUnDef{AfterBeg#1}% This is done to skip the command for environments
                     % which can take arguments, like multicols; YOU MUST NOT
                     % USE \AfterBegin{...}{...} for such environments! 
        \let\SaveBegEnd=\BeginEnvironment%
\else
        % Start this environment
                \BeginEnvironment{#1}%
        % and execute code after \begin{environment}
                \csname AfterBeg#1\endcsname
        \let\SaveBegEnd=\NullCom
\fi
\SaveBegEnd{#1}
}
\def\end#1{%
% execute code before \end{environment}
\IfUnDef{BeforeEnd#1}\else\csname BeforeEnd#1\endcsname\fi%
% close this environment
\EndEnvironment{#1}
% and execute code after \begin{environment}
\IfUnDef{AfterEnd#1}\else\csname AfterEnd#1\endcsname\fi%
}
\long\def\BeforeBegin#1#2{\expandafter\gdef\csname BeforeBeg#1\endcsname {#2}}
\long\def\BeforeEnd  #1#2{\expandafter\gdef\csname BeforeEnd#1\endcsname {#2}}
\long\def\AfterBegin #1#2{\expandafter\gdef\csname AfterBeg#1\endcsname {#2}}
\long\def\AfterEnd   #1#2{\expandafter\gdef\csname AfterEnd#1\endcsname{#2}}

%%%% end of atbeginend.sty

%  ****************************************
%  *            PAGE LAYOUT               *
%  ****************************************

% Page size, spacing parameters, etc. 

\textwidth=7.1in
\columnsep=0.3125in
\parindent=0.125in
\voffset=-20mm
\hoffset=-7.5mm

\topmargin=0in
\headheight=.15in
\headsep=0.5in
\oddsidemargin=0in
\evensidemargin=0in
\parskip=0cm

\tolerance=600          % 3x "normal" value; cuts down on overfull complaints

%% AV,MM, to have 64 lines per column, with textheight 25cm:
\textheight=64\baselineskip
\textheight=\baselinestretch\textheight
\ifdim\textheight>25.2cm\textheight=25.0cm\fi

%% so that the margin at the bottom of the page is about equal to margin at top. -- AH
\advance\textheight -54pt

\topskip\baselineskip
\maxdepth\baselineskip



\def\eqsecnum{
    \@newctr{equation}[section]
    \def\theequation{\hbox{\normalsize\arabic{section}-\arabic{equation}}}}

%% running heads:
\def\lefthead#1{\gdef\@versohead{#1}} \lefthead{\relax}
\def\righthead#1{\ifanonymous\gdef\@rectohead{Anonymous author(s)}\else\gdef\@rectohead{#1}\fi} \righthead{\relax}
\let\shorttitle\lefthead
\let\shortauthors\righthead

\def\ps@apjheads{\let\@mkboth\markboth
     \def\@evenfoot{}
    \def\@evenhead{\lower9mm\hbox to\textwidth{%
                     \rm\thepage\hfill\textsc{\@rectohead}\hfill}}}
    \def\@oddfoot{}
    \def\@oddhead{\lower9mm\hbox to\textwidth{
                     \hfil\rm\textsc{\@versohead}\hfil \rm\thepage}}

\pagestyle{apjheads}

\@twosidetrue

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    TITLE PAGE                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% October 2017, adding preprint number capability
\newcount\rptnum
\newcount\rptloopnum
\def\reportnum#1{\global\advance\rptnum by 1
\expandafter\gdef\csname report\the\rptnum\endcsname{\hbox to
\textwidth{\hskip.75\textwidth\relax#1\relax\hfill}}}

\long\def\frontmatter@title@above{
\ifrnaas
\vspace*{-\headsep}\vspace*{\headheight}
\footnotesize
\noindent{\sc Draft version \today}\\[2pt]
{\footnotesize Typeset using \LaTeX\ {\bf RNAAS} style in AASTeX631}
\par\vspace*{-\baselineskip}\vspace*{0.625in}
\else
\vbox to 0pt{\vskip-48pt\normalsize\rptloopnum=0\loop
\global\advance\rptloopnum by 1
\expandafter\ifx\csname report\the\rptloopnum\endcsname\relax
\else
\csname report\the\rptloopnum\endcsname
\vskip1pt
\repeat
\vss
}
\ifmodern
\vglue-18pt
{\footnotesize
\raggedright
{\sc Draft version \today}\\[2pt]
{\footnotesize
Typeset using \LaTeX\ {\bf modern} style in AASTeX631}
\vskip48pt
}
\else
%%
\ifpreprint
\vspace*{-\headsep}\vspace*{\headheight}
\footnotesize
\noindent{\sc Draft version \today}\\[2pt]
{\footnotesize Typeset using \LaTeX\ {\bf preprint} style in AASTeX631}
\par\vspace*{-\baselineskip}\vspace*{0.625in}
\else
%%
\ifpreprinttwo
\vspace*{-\headsep}\vspace*{\headheight}
\footnotesize
{\footnotesize\textsc{\@journalinfo}}\par
{\footnotesize Typeset using \LaTeX\ {\bf preprint2} style in
AASTeX631}
\par\vspace*{-\baselineskip}\vspace*{0.625in}
\else
%%
\iftwocolstyle
\vspace*{-\headsep}\vspace*{\headheight}
\footnotesize
{\footnotesize\textsc{\@journalinfo}}\par
{\footnotesize Typeset using \LaTeX\ {\bf twocolumn} style in
AASTeX631}
\par\vspace*{-\baselineskip}\vspace*{0.625in}
\else
\ifmanu
\vspace*{-\headsep}\vspace*{\headheight}
\footnotesize
\noindent\textsc{\@journalinfo}\\[-8pt]
{\footnotesize Typeset using \LaTeX\ {\bf manuscript} style in
AASTeX631}
\par\vspace*{-\baselineskip}\vspace*{0.625in}
\else
%% Default
\vspace*{-\headsep}\vspace*{\headheight}
\footnotesize
{\footnotesize\textsc{\@journalinfo}}\par
{\footnotesize Typeset using \LaTeX\ default style in AASTeX631}%% default style
\par\vspace*{-\baselineskip}\vspace*{0.625in}
\fi\fi\fi\fi\fi
%%
\fi %% ends test to see if rnaas option was used; if rnaas is true
    %% none of this command will be activated.
}%

\def\frontmatter@title@produce{%
 \begingroup
\ifnumlines\nolinenumbers\fi
 \frontmatter@title@above
  \frontmatter@title@format
  \@title
%% October 2017, was +12pt
  \@ifx{\@title@aux\@title@aux@cleared}{}{%
\ifanonymous\else%% added may 20, 2020
  \expandafter\frontmatter@footnote\expandafter{\@title@aux}%
\fi
  }%
  \par
%% \class@warn{Unused short title ignored}%
\ifmodern
\vskip-42pt
\else
\ifmanu
\vskip-60pt
\else
  \frontmatter@title@below
\fi
\fi
 \endgroup
}%

\newcount\fncount
\newcount\endfncount
\long\def\tempfootnote#1{\global\advance\fncount by 1\relax%
\expandafter\gdef\csname
footnote\the\fncount\endcsname{\if@two@col\hsize=.5\textwidth
\advance\hsize by -18pt
\fi\relax#1}}

\def\tempfootnoteintitle#1{}

\long\def\ltx@foottext#1#2{%
 \begingroup
\expandafter\ltx@make@current@footnote\expandafter{\@mpfn}{#1}%
\@footnotetext{\vtop{\iftwocolstyle\hsize=.5\textwidth
\advance\hsize-18pt
\fi #2\vskip2pt}}% fixed for two col style, march 15, 2019
 \endgroup
}%


\def\kickoutfootnotes{\c@footnote=0\relax%
\loop\advance\endfncount by 1\relax%
\expandafter\ifx\csname footnote\the\endfncount\endcsname\relax%
\else%
\def\thempfn{\fnsymbol{footnote}}%
\advance\c@footnote by 1\relax%
\unskip\footnote{\csname footnote\the\endfncount\endcsname}%
\repeat}

%% 
\renewcommand\title[2][]{%
\def\@title{\setbox0=\vbox{\let\footnote\tempfootnote #2}%
{\let\footnote\tempfootnoteintitle%\uppercase
{#2}}%% right place for kickoutfootnotes below
\kickoutfootnotes
\ifdoublespace\vskip 3\baselineskip\fi
}%
 \def\@shorttitle{#1}%
 \let\@AF@join\@title@join
}%

\def\frontmatter@title@format{
%% No exception for rnaas, May, 2019
%\ifrnaas
%\large\scshape\centering
%\else
\normalsize
%% added Mar 2019
\bf\centering
%\fi
}

\def\frontmatter@title@below{\vskip8pt}%


\def\frontmatter@authorformat{\small\advance\baselineskip-3pt
\parskip=0pt\ifrnaas\else\scshape\fi
\ifmodern
\vskip48pt
\centering
\leftskip=.05in plus 1fil
\rightskip=.05in plus 1 fil
\baselineskip=13pt
\else
\centering
\fi
}%


\def\frontmatter@affiliationfont{\normalfont\footnotesize
%% changed october 2017
\it
\iflongauthor\else
\ifrnaas\else
\rightskip-12pt plus 1fil
\leftskip6pt \parindent-4pt
\fi\fi
}%

 \def\frontmatter@finalspace{\ifrnaas\vspace*{-\baselineskip}\vspace*{0.0in}
\else
\vskip-24pt\fi}

\def\@separator{\\}%

\newcounter{front@matter@foot@note}\setcounter{front@matter@foot@note}{0}

\iflongauthor%
\renewcommand\affiliation[1]{%
\parskip=2pt
\def\@centercr{\vrule depth 3pt width0pt\vskip1sp}
 \move@AU\move@AF%
\iflongauthor\let\@listand\relax\fi
 \begingroup%
  \@affiliation{%\hspace*{2mm}
\ifnum\countauthors<\AuthorCollaborationLimit
 #1\ifmodern\iflongauthor\baselineskip=12pt\else\vskip2pt\fi\else\baselineskip=11pt
\fi\fi
\ifnum\countauthors=\AuthorCollaborationLimit
 #1\ifmodern\iflongauthor\baselineskip=12pt\else\vskip2pt\fi\else\baselineskip=11pt
\fi\fi
\iflongauthor \vskip-10pt\ifnum\countauthors>\AuthorCollaborationLimit\iftwelvepoint\vskip-3pt\else\iftwocolstyle\else\vskip-6pt\fi\fi\fi%
\fi
}%
\let\@centercr\savecentercr}%
\else
%% older version
\renewcommand\affiliation[1]{%
 \move@AU\move@AF%
 \begingroup%
  \@affiliation{%\hspace*{2mm}
%% october 2017 changes:
\centering
 #1\ifmodern
\baselineskip=9.5pt
 \else
        \iftwelvepoint
	\ifpreprint\baselineskip=12pt\else\ifpreprinttwo\else\baselineskip19pt\fi\fi\else \baselineskip=9.5pt\fi
\fi}%
}%
\fi %% end switch for longauthor

%   ABSTRACT
\def\frontmatter@abstractfont{\normalsize\parindent=9pt
}%
%% october 2017
\def\frontmatter@abstractwidth{478pt}
\def\frontmatter@preabstractspace{12pt}
\def\frontmatter@postabstractspace{12pt}

\def\abstractname{ABSTRACT}
\long\def\frontmatter@abstractheading{%
 \begingroup
  \centering
\ifmodern\else\hskip34pt \fi \abstractname
  \vskip 1mm
  \par
 \endgroup
}%

\newif\ifabstract%
\renewenvironment{frontmatter@abstract}{%
  \aftermaketitle@chk{\begin{abstract}}%
\global\abstracttrue
  \global\setbox\absbox\vbox\bgroup
   \color@begingroup
%% width of abstract changed oct 2017
\ifmodern
 \columnwidth\textwidth
 \hsize\columnwidth
\else
\hsize = 478pt
\fi
   \@parboxrestore
   \def\@mpfn{mpfootnote}\def\thempfn{\thempfootnote)}\c@mpfootnote\z@ % should be footnote, not mpfootnote?
%   \let\@footnotetext\frontmatter@footnotetext
   \minipagefootnote@init
   \let\set@listindent\set@listindent@
   \let\@listdepth\@mplistdepth \@mplistdepth\z@
   \let@environment{description}{frontmatter@description}%
   \@minipagerestore
   \@setminipage
    \frontmatter@abstractheading
    \frontmatter@abstractfont
%    \let\footnote\mini@note
\ifmodern\else\everypar={\leftskip=34pt}\fi
%    \expandafter\everypar\expandafter{\the\everypar\addcontents@abstract\everypar{}}%
\ifnumlines\let\go\linenumbers\else\let\go\relax\fi\go
}{%
\ifnumlines\let\go\endlinenumbers\else\let\go\relax\fi\go
    \par
    \unskip
    \minipagefootnote@here
    \@minipagefalse   %% added 24 May 89
    \color@endgroup
\egroup% end setbox\absbox
}%

\let\abstract\frontmatter@abstract

\def\frontmatter@abstract@produce{%
  \par
  \preprintsty@sw{%
   \do@output@MVL{%
    \vskip\frontmatter@preabstractspace
    \vskip200\p@\@plus1fil
    \penalty-200\relax
    \vskip-200\p@\@plus-1fil
   }%
  }{%
   \addvspace{\frontmatter@preabstractspace}%
  }%
   \begingroup
    \dimen@\baselineskip
    \setbox\z@\vtop{\unvcopy\absbox}%
    \advance\dimen@-\ht\z@\advance\dimen@-\prevdepth
    \@ifdim{\dimen@>\z@}{\vskip\dimen@}{}%
   \endgroup
   \begingroup
   \prep@absbox
%% centered, oct 2017
\unvbox\absbox
    \post@absbox
   \endgroup
  \@ifx{\@empty\mini@notes}{}{\mini@notes\par}%
  \addvspace\frontmatter@postabstractspace
\global\abstractfalse
\vskip12pt
}%

\newif\ifbib

%  ****************************************
%  *             KEYWORDS                 *
%  ****************************************


  \def\@keys@name{\textit{Keywords:}\/~\mbox{}}%

\newlength{\keys@width}
\def\frontmatter@keys@format{\ifmodern\vskip0pt\else\vspace*{0.5mm}\fi%
  \settowidth{\keys@width}{\normalsize\@keys@name}%
\ifmodern\else
\rightskip=0.5in
\leftskip=34pt
\fi
\parindent=0pt%
    \hangindent=\keys@width\hangafter=1\normalsize}%
\def\@keywords@produce#1{%
 \showKEYS@sw{%
  \begingroup%
   \frontmatter@keys@format%
   \@keys@name#1
\vrule depth 12pt width 0pt
\ifnobreakafterkeywords
\vrule depth 24pt width0pt\fi
\par
  \endgroup
 }{%
  \@if@empty{#1}{}{%
   \class@warn{If you want your keywords to appear in your output, use document class option showkeys}%
  }%
 }%
}%

\let\subjectheadings=\keywords


%  ****************************************
%  *             FOOTNOTES                *
%  ****************************************

%% legacy notes and footnote code

% Footnotes on the last page: user issues \lastpagefootnotes. It catches all
% footnotes and issues them before \begin{refernces} or
% \begin{\thebibliography} or \begin{appendix}, whichever comes first,
% at the end of right column.  
\newcounter{lastfootnote}
\let\orig@footnote=\footnote
%
\def\spit@out@footnotes@{\addtocounter{footnote}{-\c@lastfootnote}\vspace*{-\baselineskip}\vspace*{\skip\footins}\bgroup\footnotesize\lastfootnote\par\egroup\let\footnote=\orig@footnote}
%
\let\spit@out@footnotes\relax
%
\long\def\lastpagefootnotes{%
\raggedbottom% Have to use flushbottom with revtex4-1 but it eats away
             % last page footnotes
\setcounter{lastfootnote}{0}
\long\gdef\lastfootnote{\mbox{}}
\let\spit@out@footnotes\spit@out@footnotes@
\long\def\footnote##1{\refstepcounter{lastfootnote}\footnotemark\g@addto@macro\lastfootnote{\\[\footnotesep]\refstepcounter{footnote}\mbox{}\hspace*{3mm}\textsuperscript{\thefootnote}~##1}}
% We need to mess with footnotes in appendix only if it is in a
% different mode (twocol vs onecol) than the main text
\def\mess@with@appendix@footnotes{
\BeforeBegin{appendix}{\spit@out@footnotes\let\spit@out@footnotes\relax}
\BeforeEnd{appendix}{\spit@out@footnotes\let\spit@out@footnotes\relax}
\AfterEnd{appendix}{\let\footnote=\orig@footnote}
}

\if@two@col
  \if@two@col@app\else
    \mess@with@appendix@footnotes
  \fi
\fi

\BeforeBegin{references}{\spit@out@footnotes\let\spit@out@footnotes\relax}
\BeforeBegin{thebibliography}{\spit@out@footnotes\let\spit@out@footnotes\relax}
% in case \lastpagefootnotes were given inside \begin{appendix}\end{appendix},
\AfterEnd{references}{\let\footnote=\orig@footnote}
\AfterEnd{thebibliography}{\let\footnote=\orig@footnote}
}
\let\lastpagefootnote=\lastpagefootnotes

%% probably not needed:
\def\notetoeditor#1{}%   % We do not need notes to editor in the preprint
\def\placetable#1{}%   % We do not need notes to editor in the preprint
\def\placefigure#1{}%   % We do not need notes to editor in the preprint

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\ifdoublespace %used for manuscript
\footnotesep 12pt
\skip\footins 36pt
\else
\footnotesep 12pt
\skip\footins 24pt
\fi

\let\footnoterule\relax

\def\@makefntext#1{\hsize=\columnwidth\mbox{}\hspace*{3mm}\@makefnmark~#1}


%  ****************************************
%  *             SECTIONS                 *
%  ****************************************

\setcounter{secnumdepth}{3}

\newcount\tempsectnum

\newif\if@firstsection \@firstsectiontrue


\def\section{%
\ifanonymous\nocollaboration{0}\fi
\if@firstsection 
%% In case no collaboration is listed:
\ifcollaborationon\else\let\doauthor\olddoauthor
\ifanonymous \else
\let\allauthors=\oldallauthors\fi
\fi
%\edef\currauthorlimit{\the\AuthorCollaborationLimit}
%\collaboration{\currauthorlimit}{}
     \maketitle
\global\@firstsectionfalse
     \setcounter{footnote}{\thefront@matter@foot@note}%
     \let\footnotetext=\old@foot@note@text
     \let\footnotemark=\old@foot@note@mark
\ifnobreakafterkeywords
\ifmodern\vskip24pt
\else
\vskip12pt\fi%%
\goodbreak
\else
\ifrnaas\else
 \clearpage\fi
\fi
   \if@two@col
   \twocolumngrid
   \fi
\fi
\tempsectnum=\the\c@section
\advance\tempsectnum by 1
\xdef\cref@currentlabel{section \the\tempsectnum}
    \@startsection{section}{1}{\z@}{9pt plus 1pt minus
    1pt}{4pt}{\apjsecfont\center}} 

\def\subsection{\@startsection{subsection}{2}{\z@}{9pt plus 1pt minus 1pt}{4pt}%
  {\normalsize\itshape \center}}

\def\subsubsection{\@startsection{subsubsection}{3}{\z@}%
  {2ex plus 1ex minus .2ex}{1ex plus .2ex}{\small\itshape \center}}

\def\paragraph{\@startsection{paragraph}{4}{\z@}%
  {1.5ex plus 1ex minus .2ex}{0pt}{\small\itshape}}

\def\subparagraph{\@startsection{subparagraph}{5}{\z@}%
  {1ex plus 1ex minus .2ex}{-0.5\parindent}{\small\itshape}}


\def\thesection{\arabic{section}}
\def\thesubsection{\thesection.\arabic{subsection}}
\def\thesubsubsection{\thesubsection.\arabic{subsubsection}}
\def\theparagraph{\thesubsubsection.\arabic{paragraph}}
\def\p@section        {}
\def\p@subsection     {}
\def\p@subsubsection  {}
\def\p@paragraph      {}
\def\p@subparagraph   {}


\def\sec@upcase#1{\uppercase{#1}}
\def\subsec@upcase#1{\relax{#1}}

%
% How the section number will appear in the section title - AV
\def\ApjSectionMarkInTitle#1{\ifrnaas\else #1.\ \fi}
\def\ApjSectionpenalty{0}


\def\@sect#1#2#3#4#5#6[#7]#8%
{\ifnum#2=1\setbox0=\hbox{\def\label##1{\gdef\templabel{##1}}#7}\fi
\@tempskipa #5\relax 
 \ifdim \@tempskipa >\z@ \begingroup
     #6\relax 
  \ifnum #2>\c@secnumdepth \def \@svsec {}\else 
    \refstepcounter{#1} \edef \@svsec {\ApjSectionMarkInTitle 
    {\csname the#1\endcsname}}\fi
  \@hangfrom {\hskip #3\relax 
    \ifnum #2=1{\secnum@size {\rm\@svsec~}}%
    \else {\subsecnum@size {\rm\@svsec~}}\fi }%
  {\interlinepenalty \@M 
   \ifnum #2=1\sec@upcase{#8}%
   \else \subsec@upcase{#8}\fi\par}\endgroup 
  \csname #1mark\endcsname {#7}\addcontentsline{toc}{#1}%
  {\ifnum #2>\c@secnumdepth \else \protect \numberline 
     {\csname the#1\endcsname }\fi #7}%
  \else 
  \ifnum #2>\c@secnumdepth \def \@svsec {}\else 
    \refstepcounter{#1} \edef \@svsec {\ApjSectionMarkInTitle 
    {\csname the#1\endcsname}}\fi
    \def \@svsechd {#6\hskip #3%
    \ifnum #2=1{\secnum@size{\rm\@svsec~}}\else{\subsecnum@size{\rm\@svsec~}}\fi%
    \ifnum #2=1\sec@upcase{#8}\else\subsec@upcase{#8}\fi%
    \ifnum #2=4\hskip 0.4ex{\rm ---}\fi%
     \csname #1mark\endcsname {#7}\addcontentsline{toc}{#1}%
      {\ifnum #2>\c@secnumdepth \else \protect \numberline {\csname
      the#1\endcsname }\fi #7}}\fi \@xsect {#5} \penalty \ApjSectionpenalty
%-refstepcounter is now within a group. So \@currentlabel, which is normally
%-set by \refstepcounter is hidden within a group. Need to set it manually. 
\protected@edef\@currentlabel{\csname p@#1\endcsname\csname
the#1\endcsname} 
\expandafter\ifx\csname templabel\endcsname\relax
\let\go\relax
\else
\def\go{\label{\templabel}}\fi\go
\let\templabel\relax
}

\def\@ssect#1#2#3#4#5{\@tempskipa #3\relax
   \ifdim \@tempskipa>\z@
     \begingroup #4\@hangfrom{\hskip #1}{\interlinepenalty \@M
       \sec@upcase{#5}\par}\endgroup
   \else \def\@svsechd{#4\hskip #1\relax \sec@upcase{#5}}\fi
    \@xsect{#3}
% MM:
\penalty \ApjSectionpenalty}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% changed to begin...end form, May 20,2020, AH
%% Acknowledgments, use \begin{acknowledgments}...\end{acknowledgments}
%% or \begin{acknowledgements}...\end{acknowledgements}

{\catcode`#=12
\gdef\hashmark{#}}

\def\xacknowledgments{acknowledgments}
\def\xacknowledgements{acknowledgements}
\newbox\ackbox

\def\acknowledgments{%
\ifx\@currenvir\xacknowledgments\let\go\relax\else
\let\go\stop
\typeout{x^^J^^J^^J^^J^^J^^J^^J^^J
 ===================^^J
In AASTeX v6.3.1 the \string\acknowledgments\space command has been deprecated.^^J^^J
Instead, please use the begin/end form:^^J^^J
"\string\begin\string{acknowledgments\string}...\string\end\string{acknowledgments\string}"^^J^^J 
when using acknowledgments.^^J^^J See here:
\url{https://journals.aas.org/aastexguide/\hashmark acknowledgments}
^^J^^J
 for more
details.^^J
 ===================^^J
}%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Printed error
\vskip12pt
\hrule height 1pt
\vskip12pt
\noindent ERROR:
In AASTeX v6.3.1 the {\tt\string\acknowledgments} command has been
deprecated.\\
Instead, please use the begin/end form:\\
{\tt
\string\begin\string{acknowledgments\string}...\string\end\string{acknowledgments\string}}\\
when using acknowledgments. For more details, see:\\ 
\url{https://journals.aas.org/aastexguide/\hashmark acknowledgments}
%
\fi\go
%%
\global\setbox\ackbox=\vbox\bgroup
\expandafter\ifx\csname internallinenumbers\endcsname\relax\else
\begin{internallinenumbers}
\fi
\vskip 5.8mm plus 1mm minus 1mm
\vskip1sp
%\section*{Acknowledgments}
\noindent\ignorespaces}

\def\endacknowledgments{
\expandafter\ifx\csname internallinenumbers\endcsname\relax\else
\end{internallinenumbers}
\fi
\egroup%% completes ackbox
\ifanonymous
\vskip 5.8mm plus 1mm minus 1mm
\vskip1sp
%\section*{Acknowledgments}
\centerline{(Acknowledgments anonymized for review)}
\else
\vbox{\unvbox\ackbox}
\fi\vskip6pt}

%% alternate spelling
\def\acknowledgements{%
\ifx\@currenvir\xacknowledgements\let\go\relax\else
\let\go\stop
\typeout{x^^J^^J^^J^^J^^J^^J^^J^^J
 ===================^^J
In AASTeX v6.3.1 the \string\acknowledgements\space command has been deprecated.^^J^^J
Please use the begin/end form:^^J^^J
"\string\begin\string{acknowledgements\string}...\string\end\string{acknowledgements\string}"^^J^^J 
when using acknowledgements.^^J^^J See here:
\href{https://journals.aas.org/aastexguide/\hashmark acknowledgments}
{https://journals.aas.org/aastexguide/\hashmark acknowledgments}
^^J^^J
 for more
details.^^J
 ===================^^J
}%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Printed error
\vskip12pt
\hrule height 1pt
\vskip12pt
\noindent ERROR:
In AASTeX v6.3.1 the {\tt\string\acknowledgements} command has been
deprecated.\\
Instead, please use the begin/end form:\\
{\tt
\string\begin\string{acknowledgements\string}...\string\end\string{acknowledgements\string}}\\
For more details, see:\\ 
\url{https://journals.aas.org/aastexguide/\hashmark acknowledgments}
%
\fi\go
%%
\global\setbox\ackbox=\vbox\bgroup
\expandafter\ifx\csname internallinenumbers\endcsname\relax\else
\begin{internallinenumbers}
\fi
\vskip 5.8mm plus 1mm minus 1mm
\vskip1sp
%\section*{Acknowledgments}
\noindent\ignorespaces}

\def\endacknowledgements{
\expandafter\ifx\csname internallinenumbers\endcsname\relax\else
\end{internallinenumbers}
\fi
\egroup%% completes ackbox
\ifanonymous
\vskip 5.8mm plus 1mm minus 1mm
\vskip1sp
%\section*{Acknowledgements}
\centerline{(Acknowledgements anonymized for review)}
\else
\vbox{\unvbox\ackbox}
\fi\vskip6pt}






%  ****************************************
%  *          APPENDIX                    *
%  ****************************************

\newcounter{remember@figure@num}
\newcounter{remember@table@num}

% this was an environment earlier, which doesn't make sense since we don't
% do \begin{appendix}...\end{appendix}. Changed to \appendix which is how it is used.

%   numberedappendix: Needed to label appendix sections A, B, .
%   appendixfloats: Needed. ??

\newif\ifappendixon
\def\appendix{\global\appendixontrue
\if@two@col  
\onecolumngrid
\noindent\mbox{}\vrule height 24pt width0pt\hfill{\apjsecfont APPENDIX}\hfill\mbox{}\par
\vskip18pt
   \if@two@col@app\global\@two@coltrue\twocolumngrid \fi
   % above, we want onecolumngrid to be default. Only twocolumn is asked for in documentclass option
\else
\noindent\mbox{}\vrule height 24pt width0pt\hfill{\apjsecfont
APPENDIX}\hfill\mbox{}\par 
\vskip18pt
    \if@two@col@app\global\@two@coltrue\twocolumngrid 
         \fi\fi
% \vrule used for extra space; otherwise revtex4-1 sometimes eats
% away the last line before appendix
        \nopagebreak\medskip\@nobreaktrue\def\ApjSectionpenalty{\@M}
        \@firstsectionfalse
          \setcounter{section}{0}
          \setcounter{subsection}{0}
          \setcounter{equation}{0}
%%
\ifletteredappendix
          \def\thesection{\Alph{section}}
          \def\theequation{\hbox{\Alph{section}\arabic{equation}}}
%% these are NOT supposed to reset to zero
%\setcounter{table}{0}
%\setcounter{figure}{0}
\fi
%%
\if@number@appendix@floats
% we don't want to set equations to zero in appendix, because there
% might then be two equation 1's, etc., confusing for cross referencing.
%\setcounter{equation}{0}
          \def\thesection{\Alph{section}}
          \def\theequation{\hbox{\Alph{section}\arabic{equation}}}
          \def\section{\@startsection {section}{1}{\z@} 
            {10pt}{5pt}{\centering\scshape\apjsecfont}}
\else
% Do not use appendix numbers in the titles
          \def\ApjSectionMarkInTitle{\AppendixApjSectionMarkInTitle}
\fi
\ifappletter
\let\savesection\section
\def\section{\resetapptablenumbers\savesection}
\fi
}
%


%%

%  ****************************************
%  *          BIBLIOGRAPHY                *
%  ****************************************

\renewenvironment{references}[0]{
  \onecolumngrid
  \par
  \vspace{10pt plus 3pt}
  \noindent \makebox[\textwidth][c]{\small\scshape REFERENCES}
  \par
  \vspace*{4pt plus 3pt}
  \set@column@hsize{2}\twocolumngrid 
%% Jan29, 2021, added \ifonecolstyle, \ifpreprinttwo
\ifnumlines
\ifonecolstyle
\ifpreprinttwo\else
\advance\linenumbersep-12pt\fi\fi\fi
  \parindent=0cm \parskip=0cm
  \def\baselinestretch{1.0}
  \footnotesize \baselineskip=\baselinestretch\bibbaselineskip plus 1pt
  minus 1pt \interlinepenalty \@M
  \hyphenpenalty=10000
  \frenchspacing    % AV - to get right spacing after ``et al.'' 
  \def\refpar{\par\hangindent=1.2em\hangafter=1}}
{
  \onecolumngrid
}

% redefine thebibliography

% remove numbers from the reference list
\def\@biblabel#1{\relax}

\newskip\bibskip
\bibskip=0pt plus 1pt
\let\savebibitem\bibitem
\def\bibitem{\vskip\bibskip\savebibitem}
\newdimen\bibindent
\renewenvironment{thebibliography}[1]{\global\bibtrue
%%\ifrnaas\newpage\fi% Not wanted, March 2019
\onecolumngrid
\vspace{20pt}
\goodbreak
    \hbox to\textwidth{\hss\normalsize REFERENCES\hss}
\vspace{6pt}\parskip=0pt
\twocolumngrid 
%% Jan29, 2021, added \ifonecolstyle
\ifnumlines
\ifonecolstyle
\ifpreprinttwo\else
\advance\linenumbersep-12pt\fi\fi\fi
\par
 \raggedright
\small
\ifmodern\else
 \vspace{10pt plus 3pt}\fi
\par
\topsep=0pt
 \list{}%
   {
     \parindent=0pt \parskip=1pt plus 1pt \parsep=0pt % AV
     \bibindent=0pt                          %
\ifmodern\vskip-12pt
\baselineskip=13pt plus 1pt
\else
\ifdoublespace
\baselineskip=20pt
\else
\baselineskip=13pt plus 1pt \fi\fi \interlinepenalty \@M  % AV
     \frenchspacing    % AV - to get right spacing after ``et al.'' 
     \hyphenpenalty=10000
     \itemindent=-1.0em                      %
     \itemsep=0pt                            %
     \listparindent=0pt                      %
     \settowidth\labelwidth{0pt} %
     \labelsep=0pt                           %
     \leftmargin=1.0em
     \advance\leftmargin\labelsep
%%%      \usecounter{enumiv}%
      \let\p@enumiv\@empty
      \renewcommand\theenumiv{\relax}}%
    \sloppy\clubpenalty10000\widowpenalty10000%
    \sfcode`\.\@m\relax
%%%\item[] This May be necessary, but causes extra vertical space in first column
%%%          of the bibliography.
}
  {\def\@noitemerr
    {\@latex@warning{Empty `thebibliography' environment}}%
\endlist
    \onecolumngrid % to balance references
\global\bibfalse
\newpage
  }

% %% AV: 

\def\reference{\@ifnextchar\bgroup {\@reference}
        {\@latexerr{Missing key on reference command}
        {Each reference command should have a key corresponding to a markcite somewhere in the text}}}
\def\@reference#1{\refpar}

%% subtitle header and journal info, legacy, probably not needed
% \def\submitted#1{\gdef\@submitted{#1}}
% \let\slugcomment\submitted   % slugcomment == submitted
% \submitted{Draft version \today}
% \journalinfo{\@submitted}

 \def\journalinfo#1{\gdef\@journalinfo{#1}}
 \journalinfo{Draft version \today}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Equations                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EQNARRAY with reduced spacing around tab characters - AV
\def\eqnarray{%
  \stepcounter{equation}%
  \def\@currentlabel{\p@equation\theequation}%
  \global\@eqnswtrue
  \setlength{\arraycolsep}{0.25\arraycolsep}
  \m@th
  \global\@eqcnt\z@
  \tabskip\@centering
  \let\\\@eqncr
  $$\everycr{}\halign to\displaywidth\bgroup
  \hskip\@centering$\displaystyle\tabskip\z@skip{##}$\@eqnsel
  &\global\@eqcnt\@ne\hskip \tw@\arraycolsep \hfil${##}$\hfil
  &\global\@eqcnt\tw@ \hskip \tw@\arraycolsep
  $\displaystyle{##}$\hfil\tabskip\@centering
  &\global\@eqcnt\thr@@ \hb@xt@\z@\bgroup\hss##\egroup
  \tabskip\z@skip
  \cr
  }

%  ****************************************
%  *         TABLES AND FIGURES           *
%  ****************************************

\def\@arstrut@hline@clnc{0.5\p@}% % Klootch: magic number

\setcounter{topnumber}{7}

\setlength{\belowcaptionskip}{4pt}% 
\setlength{\abovecaptionskip}{5pt}

\setlength{\textfloatsep}{9pt}
\setlength{\dbltextfloatsep}{0pt}

%  \addtolength{\textfloatsep}{17mm plus 10mm minus 10mm}
 % \addtolength{\dbltextfloatsep}{7mm plus 10mm minus 10mm}


\renewcommand{\topfraction}{1.0}
\renewcommand{\bottomfraction}{1.0}
\renewcommand{\textfraction}{0.0}
\renewcommand{\dbltopfraction}{0.85}
\renewcommand{\dblfloatpagefraction}{0.85}
\setcounter{topnumber}{9}
\setcounter{bottomnumber}{9}
\setcounter{totalnumber}{20}
\setcounter{dbltopnumber}{9} 

% Copied from revtex4.cls; without it, captions are centered
\def\@xfloat@prep{%
  \let\footnote\footnote@latex
  \def\@mpfn{mpfootnote}%
  \def\thempfn{\thempfootnote}%
  \c@mpfootnote\z@
  \let\@footnotetext\@mpfootnotetext
  \let\H@@footnotetext\@mpfootnotetext
  \let\@makefntext\@mpmakefntext
}%


%%%%%%%%%%%% \cutinhead

%% Number of \pt@ncol set when \tablecolumns{} used at beginning of
%% deluxetable.

\def\@ptabularcr{\\}
\newcommand\cutinhead[1]{% 
 \noalign{\vskip 1.5ex}% 
 \hline 
 \@ptabularcr 
 \noalign{\vskip -1.5ex}% 
 \multicolumn{\pt@ncol}{c}{#1}% 
 \@ptabularcr 
 \noalign{\vskip .8ex}% 
 \hline 
 \@ptabularcr 
 \noalign{\vskip -2ex}% 
}% 

\newcommand\sidehead[1]{% 
 \noalign{\vskip 1.5ex}% 
%  \hline 
%  \@ptabularcr 
%  \noalign{\vskip -1.5ex}% 
 \multicolumn{\pt@ncol}{l}{#1}% 
 \@ptabularcr 
 \noalign{\vskip .8ex}% 
%  \hline 
%  \@ptabularcr 
%  \noalign{\vskip -2ex}% 
}% 


%% these don't seem to be relevant:
\newcommand\figcaption{\@testopt{\@xfigcaption}{}}% 
%% suggested by Joern Wilms
\def\@figcaption#1{{\def\@captype{figure}\caption{\footnotesize #1}}}
\def\@xfigcaption[#1]#2{{\def\@captype{figure}\caption{\footnotesize
#2}}}

\def\thefigure{\@arabic\c@figure}

\def\fnum@figure{{\footnotesize{\@eapj@cap@font\rm
\mbox{\@eapj@figname~\thefigure}% <== was period here
}}}

\def\thetable{\@arabic\c@table}
%%% Table captions without making a floating table
\def\tabcaption{\@ifnextchar[{\@xtabcaption}{\@tabcaption}}
\def\@tabcaption#1{{\def\@captype{table}\caption{#1}}}
\def\@xtabcaption[#1]#2{{\def\@captype{table}\caption{#2}}}

%% redefined below
\def\fnum@table{{\@eapj@cap@font \@eapj@tabname~\thetable}}

 \let\fnum@ptable=\fnum@table
 \def\fnum@ptablecont{{\centering{\scshape Table~\thetable}---{\itshape
 Continued}}}% 
 \long\def\@make@caption@text#1#2{% 
   {\small\centering#1{\scshape #2}\par\vskip1.4ex}
 }% 

\long\def\@makecaption@plano@cont#1#2{% 
  {\small \centering#1\par}\vskip1.4ex\relax
}% 


%% Commands from aastex.cls:

% This has been redefined below
%\newcommand\dataset{\@testopt\@dataset{[}}%
%\def\@dataset[#1]#2{#2}%

\newcommand\facility{\@testopt\@facility{[}}%
\def\@facility[#1]#2{{\vskip6pt{\large\it Facility:} #2}}%

\newcommand\facilities{\@testopt\@facilities{[}}%
\def\@facilities[#1]#2{{\vskip6pt{\large\it Facilities:} #2}}%


\long\def\software{\bgroup\@testopt \@software {[}}
\def\@software[#1]#2{\vskip 6pt{
\frenchspacing
\iftwelvepoint
\font\foo=cmr12
\else
\font\foo=cmr10\fi
%%
\fontdimen2\foo=3pt %% Changed from 1.5pt to 3pt, March12, 2019
%%
{\large \it Software: }
#2
%% returning to original fontdimen
\iftwelvepoint
\fontdimen2\foo=3.91663pt
\else
\fontdimen2\foo=3.33333pt
\fi
}\egroup}





\newcommand\object{\@testopt\@object{[}}%
\def\@object[#1]#2{#2}%
\newcommand\objectname{\@testopt\@objectname{[]}}% 
\def\@objectname[#1]#2{#2}% 
\newlength{\plot@width}
\def\eps@scaling{1.0}% 
\newcommand\epsscale[1]{\def\eps@scaling{#1}}% 
\newcommand\plotone[1]{% 
 \centering 
 \leavevmode 
 \setlength{\plot@width}{0.85\linewidth}
 \includegraphics[width={\eps@scaling\plot@width}]{#1}% 
}% 
\newcommand\plottwo[2]{% 
 \centering 
 \leavevmode 
 \setlength{\plot@width}{0.425\linewidth}
 \includegraphics[width={\eps@scaling\plot@width}]{#1}% 
 \hfil 
 \includegraphics[width={\eps@scaling\plot@width}]{#2}% 
}% 
\newcommand\plotfiddle[7]{% 
 \centering 
 \leavevmode 
 \vbox\@to#2{\rule{\z@}{#2}}% 
 \includegraphics[% 
  scale=#4, 
  angle=#3, 
  origin=c 
 ]{#1}% 
}% 
\newcommand\figurenum[1]{% 
 \def\thefigure{#1}% 
 \let\@currentlabel\thefigure 
 \addtocounter{figure}{\m@ne}% 
}% 

\newcommand\phn{\phantom{0}}% 
\newcommand\phd{\phantom{.}}% 
\newcommand\phs{\phantom{$-$}}% 
\newcommand\phm[1]{\phantom{#1}}% 
\let\la=\lesssim            % For Springer A&A compliance... 
\let\ga=\gtrsim 
\newcommand\sq{\mbox{\rlap{$\sqcap$}$\sqcup$}}% 
\newcommand\arcdeg{\mbox{$^\circ$}}% 
\newcommand\arcmin{\mbox{$^\prime$}}% 
\newcommand\arcsec{\mbox{$^{\prime\prime}$}}% 
\newcommand\fd{\mbox{$.\!\!^{\mathrm d}$}}% 
\newcommand\fh{\mbox{$.\!\!^{\mathrm h}$}}% 
\newcommand\fm{\mbox{$.\!\!^{\mathrm m}$}}% 
\newcommand\fs{\mbox{$.\!\!^{\mathrm s}$}}% 
\newcommand\fdg{\mbox{$.\!\!^\circ$}}% 
\newcommand\farcm@mss{\mbox{$.\mkern-4mu^\prime$}}% 
\let\farcm\farcm@mss 
\newcommand\farcs@mss{\mbox{$.\!\!^{\prime\prime}$}}% 
\let\farcs\farcs@mss 
\newcommand\fp{\mbox{$.\!\!^{\scriptscriptstyle\mathrm p}$}}% 
\newcommand\micron{\mbox{$\mu$m}}% 
\def\farcm@apj{% 
 \mbox{.\kern -0.7ex\raisebox{.9ex}{\scriptsize$\prime$}}% 
}% 
\def\farcs@apj{% 
 \mbox{% 
  \kern  0.13ex.% 
  \kern -0.95ex\raisebox{.9ex}{\scriptsize$\prime\prime$}% 
  \kern -0.1ex% 
 }% 
}% 
 

\newcommand\ion[2]{#1$\;${%
\ifx\@currsize\normalsize\small \else
\ifx\@currsize\small\footnotesize \else
\ifx\@currsize\footnotesize\scriptsize \else
\ifx\@currsize\scriptsize\tiny \else
\ifx\@currsize\large\normalsize \else
\ifx\@currsize\Large\large
\fi\fi\fi\fi\fi\fi
\rmfamily\@Roman{#2}}\relax}% 

\renewcommand\case[2]{\mbox{$\frac{#1}{#2}$}}% 
\renewcommand\slantfrac{\case}% 
\newcommand\onehalf{\slantfrac{1}{2}}% 
\newcommand\onethird{\slantfrac{1}{3}}% 
\newcommand\twothirds{\slantfrac{2}{3}}% 
\newcommand\onequarter{\slantfrac{1}{4}}% 
\newcommand\threequarters{\slantfrac{3}{4}}% 
\newcommand\ubvr{\mbox{$U\!BV\!R$}}%% UBVR system 
\newcommand\ub{\mbox{$U\!-\!B$}}%   % U-B 
\newcommand\bv{\mbox{$B\!-\!V$}}%   % B-V 
\renewcommand\vr{\mbox{$V\!-\!R$}}%   % V-R 
\newcommand\ur{\mbox{$U\!-\!R$}}%   % U-R 

%% need this change so that it works correctly in tables:
{\catcode`\$=\active
\gdef\nodata{ ~$\cdots$~ }}% 

\newcommand\diameter{\ooalign{\hfil/\hfil\crcr\mathhexbox20D}}% 
\newcommand\degr{\arcdeg}% 
\newcommand\Sun{\sun}% 
\newcommand\Sol{\sun}% 
\newcommand\sun{\odot}% 
\newcommand\Mercury{\astro{\char1}}% Mercury symbol, "1" 
\newcommand\Venus{\astro{\char2}}% Venus symbol, "2" 
\newcommand\Earth{\earth}% 
\newcommand\Terra{\earth}% 
\newcommand\earth{\oplus}% 
\newcommand\Mars{\astro{\char4}}% Mars symbol, "4" 
\newcommand\Jupiter{\astro{\char5}}% Jupiter symbol, "5" 
\newcommand\Saturn{\astro{\char6}}% Saturn symbol, "6" 
\newcommand\Uranus{\astro{\char7}}% Uranus symbol, "7" 
\newcommand\Neptune{\astro{\char8}}% Neptune symbol, "8" 
\newcommand\Pluto{\astro{\char9}}% Pluo symbol, "9" 
\newcommand\Moon{\astro{\char10}}% Moon symbol, "M" 
\newcommand\Luna{\Moon}% 
\newcommand\Aries{\astro{\char11}}% 
\newcommand\VEq{\Aries}% vernal equinox (Aries) 
\newcommand\Taurus{\astro{\char12}}% 
\newcommand\Gemini{\astro{\char13}}% 
\newcommand\Cancer{\astro{\char14}}% 
\newcommand\Leo{\astro{\char15}}% 
\newcommand\Virgo{\astro{\char16}}% 
\newcommand\Libra{\astro{\char17}}% 
\newcommand\AEq{\Libra}% autumnal equinox (Libra) 
\newcommand\Scorpius{\astro{\char18}}% 
\newcommand\Sagittarius{\astro{\char19}}% 
\newcommand\Capricornus{\astro{\char20}}% 
\newcommand\Aquarius{\astro{\char21}}% 
\newcommand\Pisces{\astro{\char22}}% 
 

\newcommand\sbond{\chem@bnd{\@sbnd}}%
\newcommand\dbond{\chem@bnd{\@dbnd}}%
\newcommand\tbond{\chem@bnd{\@tbnd}}%
\def\chem@bnd#1{%
 {%
  \kern.1em\relax
  \setbox\z@\hbox{M}%
  \dimen@ii.8em\relax
  \p@=.1em\relax
  \dimen@.5\ht\z@\dimen@i-\dimen@
  \advance\dimen@1.5\p@\advance\dimen@i-1.0\p@
  #1%
  \kern.1em\relax
  }%
 }%
\def\@sbnd{%
 \advance\dimen@-1.5\p@\advance\dimen@i1.5\p@
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 }%
\def\@dbnd{%
 \advance\dimen@-0.5\p@\advance\dimen@i0.5\p@
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 \advance\dimen@-1.5\p@\advance\dimen@i1.5\p@
 \hskip-\dimen@ii
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 }%
\def\@tbnd{%
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 \advance\dimen@-1.5\p@\advance\dimen@i1.5\p@
 \hskip-\dimen@ii
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 \advance\dimen@-1.5\p@\advance\dimen@i1.5\p@
 \hskip-\dimen@ii
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 }%

%\newcommand\anchor[2]{#2}% 
%\renewcommand\url{\@dblarg\@url}% 
%\def\@url[#1]{\anchor{#1}}% 

\let\jnl@style=\rmfamily 
\def\ref@jnl#1{{\jnl@style#1}}% 
\newcommand\aj{\ref@jnl{AJ}}%        % Astronomical Journal 
\newcommand\psj{\ref@jnl{PSJ}}%       % Planetary Science Journal
\newcommand\araa{\ref@jnl{ARA\&A}}%  % Annual Review of Astron and Astrophys 
\renewcommand\apj{\ref@jnl{ApJ}}%    % Astrophysical Journal 
\newcommand\apjl{\ref@jnl{ApJL}}     % Astrophysical Journal, Letters 
\newcommand\apjs{\ref@jnl{ApJS}}%    % Astrophysical Journal, Supplement 
\renewcommand\ao{\ref@jnl{ApOpt}}%   % Applied Optics 
\newcommand\apss{\ref@jnl{Ap\&SS}}%  % Astrophysics and Space Science 
\newcommand\aap{\ref@jnl{A\&A}}%     % Astronomy and Astrophysics 
\newcommand\aapr{\ref@jnl{A\&A~Rv}}%  % Astronomy and Astrophysics Reviews 
\newcommand\aaps{\ref@jnl{A\&AS}}%    % Astronomy and Astrophysics, Supplement 
\newcommand\azh{\ref@jnl{AZh}}%       % Astronomicheskii Zhurnal 
\newcommand\baas{\ref@jnl{BAAS}}%     % Bulletin of the AAS 
\newcommand\icarus{\ref@jnl{Icarus}}% % Icarus
\newcommand\jaavso{\ref@jnl{JAAVSO}}  % The Journal of the American Association of Variable Star Observers
\newcommand\jrasc{\ref@jnl{JRASC}}%   % Journal of the RAS of Canada 
\newcommand\memras{\ref@jnl{MmRAS}}%  % Memoirs of the RAS 
\newcommand\mnras{\ref@jnl{MNRAS}}%   % Monthly Notices of the RAS 
\renewcommand\pra{\ref@jnl{PhRvA}}% % Physical Review A: General Physics 
\renewcommand\prb{\ref@jnl{PhRvB}}% % Physical Review B: Solid State 
\renewcommand\prc{\ref@jnl{PhRvC}}% % Physical Review C 
\renewcommand\prd{\ref@jnl{PhRvD}}% % Physical Review D 
\renewcommand\pre{\ref@jnl{PhRvE}}% % Physical Review E 
\renewcommand\prl{\ref@jnl{PhRvL}}% % Physical Review Letters 
\newcommand\pasp{\ref@jnl{PASP}}%     % Publications of the ASP 
\newcommand\pasj{\ref@jnl{PASJ}}%     % Publications of the ASJ 
\newcommand\qjras{\ref@jnl{QJRAS}}%   % Quarterly Journal of the RAS 
\newcommand\skytel{\ref@jnl{S\&T}}%   % Sky and Telescope 
\newcommand\solphys{\ref@jnl{SoPh}}% % Solar Physics 
\newcommand\sovast{\ref@jnl{Soviet~Ast.}}% % Soviet Astronomy 
\newcommand\ssr{\ref@jnl{SSRv}}% % Space Science Reviews 
\newcommand\zap{\ref@jnl{ZA}}%       % Zeitschrift fuer Astrophysik 
\renewcommand\nat{\ref@jnl{Nature}}%  % Nature 
\newcommand\iaucirc{\ref@jnl{IAUC}}% % IAU Cirulars 
\newcommand\aplett{\ref@jnl{Astrophys.~Lett.}}%  % Astrophysics Letters 
\newcommand\apspr{\ref@jnl{Astrophys.~Space~Phys.~Res.}}% % Astrophysics Space Physics Research 
\newcommand\bain{\ref@jnl{BAN}}% % Bulletin Astronomical Institute of the Netherlands 
\newcommand\fcp{\ref@jnl{FCPh}}%   % Fundamental Cosmic Physics 
\newcommand\gca{\ref@jnl{GeoCoA}}% % Geochimica Cosmochimica Acta 
\newcommand\grl{\ref@jnl{Geophys.~Res.~Lett.}}%  % Geophysics Research Letters 
\renewcommand\jcp{\ref@jnl{JChPh}}%     % Journal of Chemical Physics 
\newcommand\jgr{\ref@jnl{J.~Geophys.~Res.}}%     % Journal of Geophysics Research 
\newcommand\jqsrt{\ref@jnl{JQSRT}}%   % Journal of Quantitiative Spectroscopy and Radiative Trasfer 
\newcommand\memsai{\ref@jnl{MmSAI}}% % Mem. Societa Astronomica Italiana 
\newcommand\nphysa{\ref@jnl{NuPhA}}%     % Nuclear Physics A 
\newcommand\physrep{\ref@jnl{PhR}}%       % Physics Reports 
\newcommand\physscr{\ref@jnl{PhyS}}%        % Physica Scripta 
\newcommand\planss{\ref@jnl{Planet.~Space~Sci.}}%  % Planetary Space Science 
\newcommand\procspie{\ref@jnl{Proc.~SPIE}}%      % Proceedings of the SPIE 

\newcommand\actaa{\ref@jnl{AcA}}%  % Acta Astronomica
\newcommand\caa{\ref@jnl{ChA\&A}}%  % Chinese Astronomy and Astrophysics
\newcommand\cjaa{\ref@jnl{ChJA\&A}}%  % Chinese Journal of Astronomy and Astrophysics
\newcommand\jcap{\ref@jnl{JCAP}}%  % Journal of Cosmology and Astroparticle Physics
\newcommand\na{\ref@jnl{NewA}}%  % New Astronomy
\newcommand\nar{\ref@jnl{NewAR}}%  % New Astronomy Review
\newcommand\pasa{\ref@jnl{PASA}}%  % Publications of the Astron. Soc. of Australia
\newcommand\rmxaa{\ref@jnl{RMxAA}}%  % Revista Mexicana de Astronomia y Astrofisica

%% added feb 9, 2016
\newcommand\maps{\ref@jnl{M\&PS}}% Meteoritics and Planetary Science
\newcommand\aas{\ref@jnl{AAS Meeting Abstracts}}% American Astronomical Society Meeting Abstracts
\newcommand\dps{\ref@jnl{AAS/DPS Meeting Abstracts}}% American Astronomical Society/Division for Planetary Sciences Meeting Abstracts



\let\astap=\aap 
\let\apjlett=\apjl 
\let\apjsupp=\apjs 
\let\applopt=\ao 

\newcommand\ulap[1]{\vbox\@to\z@{{\vss#1}}}% 
\newcommand\dlap[1]{\vbox\@to\z@{{#1\vss}}}% 


\newcounter{table@save}

%% March 25, 2019
%% Old v5.2 way, From Greg, This allows a number like 33N to be used
%% for a table, and the cross-references will work correctly
\newcommand\tablenum[1]{%
 \def\thetable{#1}%
 \xdef\@currentlabel{\thetable}
\global\advance\c@table-1\relax
}%

\let\savetablenum\tablenum

\def\tabletypesize#1{\gdef\currtabletypesize{#1}
\def\@table@type@size{#1}}%

\tabletypesize{\small}
\let\tablefontsize=\tabletypesize % for compatibility with old documents

\gdef\@tablecaption{}
\def\tablecaption#1{\gdef\@tablecaption{#1}}

\def\LT@endpbox{%
  \@finalstrut\@arstrutbox
  \egroup
  \the\LT@p@ftn
  \global\LT@p@ftn{}%
  \hfil}

%% Used for longtable
\def\LT@makecaption#1#2#3{%
  \LT@mcol\LT@cols c{\hbox to\z@{\hss\parbox[t]\LTcapwidth
{%
\xdef \@currentlabel{\thetable}
     \sbox\@tempboxa{\small #2. 
%% disable trackchanges commands here, so they aren't entered 2 times:
\let\added\relax
\let\deleted\relax
\let\replaced\relax
#3}%
     \ifdim\wd\@tempboxa>\hsize
      \small#2. #3%
     \else
       \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
     \fi
    \endgraf\vskip\baselineskip}%
  \hss}}
}%% 

\let\LT@makecaption@rtx=\LT@makecaption % to fight redefinition in Revtex-4.1

\def\xfigure{figure}
%% from book.cls/ used??
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
%% \small added to keep currtabletypesize from determining size of caption
  \sbox\@tempboxa{\small
%% disable trackchanges commands here, so they aren't entered 2 times:
\let\added\relax
\let\deleted\relax
\let\replaced\relax
{\bf #1.} #2}%
  \ifdim \wd\@tempboxa >\hsize
\small
{\bf    #1.} #2\par 
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfill\box\@tempboxa\hfill}%
  \fi
  \vskip\belowcaptionskip
}

\newdimen\@abovenoteskip
\newcommand\tablerefs[1]{\ifdim\@abovenoteskip=0pt\global\@abovenoteskip=10pt\fi
{\small\@tableref{\parfillskip\z@ plus1fil\relax #1\endgraf}}}%

\def\@tableref#1{%
 \par
 \vspace*{3ex}%
 {%\parbox{\pt@width} %%%%
 {\hskip1em\rm References. --- #1}\par}%
}%

%% march 2019, added \it to tablenotemark
\global\def\tablenotemark#1{{\normalfont\textsuperscript{\normalsize\it #1}}}
\global\def\tablenotetext#1#2{\footnotetext[#1]{\currtabletypesize\relax#2}}

%% redefined by AH below, since it wasn't working with tabular table
\global\def\tablenotetext#1#2{\vskip-8pt\vskip1sp\flushleft{\currtabletypesize
\noindent\hskip1em $^{#1}$ #2}\vskip1sp}

%% this version of \tablehead doesn't seem to be used, so set to \xyztablehead{}
\def\xyztablehead#1{\@table@not@headedfalse%
  \kill
  \caption{\\%
   \@tablecaption\gdef\@currentlabel{\thetable}(0)}
    \\\hline\hline%
  #1\vrule height 12pt depth 10pt width 0pt\relax 
\hskip\tabcolsep\\[.7ex]
  \hline\\[-1.5ex]
  \endfirsthead
  \caption[]{--- \emph{Continued}}\\
  \hline
  \hline\\[-1.7ex]
  #1\hskip\tabcolsep\\[.7ex]
  \hline\\[-1.5ex]
  \endhead
  \hline
  \endfoot%
}

\newif\if@table@not@headed

\newlength{\table@note@skip}
\setlength{\table@note@skip}{0.5ex}
\newlength{\deluxe@table@width}
\newlength{\@d@t@a}
\newcounter{deluxe@table@num}
\newdimen\LTcapwidth

%% \ignorespaces necessary ++== depth 6pt was depth 3pt, == added height 12pt, nov 2017
\def\colhead#1{\multicolumn{1}{c}{\vrule depth 6pt height 12pt width
0pt\relax#1}\ignorespaces}
\def\twocolhead#1{\multicolumn{2}{c}{\hss\vrule depth 6pt height 12pt width
0pt\relax#1\hss}\ignorespaces}
\def\nocolhead#1{\multicolumn{1}{h}{}\ignorespaces}
\def\dcolhead#1{\multicolumn{1}{c}{$\vrule depth 6pt height12pt
width0pt\relax#1$}\ignorespaces}

%\newcounter{LT@tables}
\def\tablewidth#1{%
  \ifdim#1=\z@
  \else
  \gdef\@d@t@@flag{1}
  \if@filesw\immediate\write\@auxout{%
   \gdef\expandafter\noexpand
   \csname deluxe@table@width@\romannumeral\c@LT@tables\endcsname
   {#1}}
  \fi
  \fi
}


\def\save@natural@width{%
      \ifnum\@d@t@@flag=0
        \setlength{\@d@t@a}{0pt}%
        \let\@d@t@b=\LT@entry%
        \def\LT@entry##1##2{\addtolength{\@d@t@a}{##2}%
        }%
        \expandafter\csname LT@\romannumeral\c@deluxe@table@num\endcsname
        \setlength{\@d@t@a}{-\@d@t@a}
        \tablewidth{\the\@d@t@a}
        \def\LT@entry{\@d@t@b}
      \fi
}

\def\lt@expand@linewidth@one{\setlength\LTleft{0pt}\setlength\LTright{0pt}}
\def\lt@expand@linewidth@two{@{\extracolsep{0pt plus 1filll}}}

\def\find@table@width{%
%%% set table width using aux file and command \tablewidth
    \setcounter{deluxe@table@num}{\c@LT@tables}
    \refstepcounter{deluxe@table@num}
    \expandafter\ifx\csname deluxe@table@width@\romannumeral\c@deluxe@table@num\endcsname\relax
      \def\@d@t@{0.999\linewidth}
    \else
      \edef\@d@t@{\expandafter\csname deluxe@table@width@\romannumeral\c@deluxe@table@num\endcsname}
    \fi
    \ifdim\@d@t@<\z@% then natural width is used
      \setlength{\deluxe@table@width}{-\@d@t@}
      \setlength{\LTcapwidth}{-\@d@t@}
      \def\lt@expand@linewidth{\relax}
      \def\lt@expand@linewidth@{}
    \else% we will enclose table in the minipage of the given width and make
         % longtable to span the full minipage width
      \ifdim\@d@t@>\z@\else\def\@d@t@{0.999\linewidth}\fi
      \setlength{\deluxe@table@width}{\@d@t@}
      \setlength{\LTcapwidth}{\@d@t@}
      \def\lt@expand@linewidth{\lt@expand@linewidth@one}
      \def\lt@expand@linewidth@{\lt@expand@linewidth@two}
    \fi
}

\newlength{\abovedeluxetableskip}
\newlength{\belowdeluxetableskip}
\setlength{\abovedeluxetableskip}{0pt}
\setlength{\belowdeluxetableskip}{0pt}
\setlength{\tabcolsep}{5pt}

\setlength\doublerulesep{1.5pt}
\newdimen\lastrowheight
\def\set@last@row@height{\setlength{\lastrowheight}{\ht\strutbox}\addtolength{\lastrowheight}{\dp\strutbox}\setlength{\lastrowheight}{-\arraystretch\lastrowheight}}

\let\tableline=\colrule % Revtex said: Command \tableline is obsolete; Use \colrule instead.. 

%% 
\newtoks\DT@p@ftn 
\global\def\xtablenotetext@DT#1#2{
  \edef\@tempa{\the\DT@p@ftn\noexpand\tablenotemark{#1}~}
  \global\DT@p@ftn\expandafter{\@tempa{\@table@type@size#2}\par}}%

%%% AH %%%%%%%%%%%%%%%%%%%%%
\let\savedollar$
\catcode`\$=\active
\let$\savedollar

%%

\def\resetdecimals{\global\let\zdoit\relax\global\let\ddoit\relax}

\def\tableheadfrac#1{}
\newcount\pt@column 
\newcount\pt@ncol 

\newcommand\tablecolumns[1]{% 
 \pt@column=#1\relax 
 \pt@ncol=#1\relax 
 \global\let\pt@addcol\@empty 
}% 

%% from older version, probably can delete:
\def\@tablecom#1{% 
 \vspace*{\table@note@skip}
\par 
{\parbox{\linewidth}{\hskip1em\rmfamily {\@eapj@cap@font Note}. --- #1}\par}% 
}% 
\def\@tableref#1{% 
 \vspace*{\table@note@skip}
\par 
{\parbox{\linewidth}{\hskip1em\rmfamily {\@eapj@cap@font References}. --- #1}\par}% 
}% 
\def\spew@tblnotes{% 
 \@ifx@empty\tblref@list{}{% 
  \@tablenotes{\tblref@list}%
  \vspace*{\table@note@skip}%
  \global\let\tblref@list\@empty 
 }% 
 \@ifx@empty\tblnote@list{}{% 
  \@tablenotes{\tblnote@list}% 
  \vspace*{\table@note@skip}%
  \global\let\tblnote@list\@empty
 }
 \the\DT@p@ftn%
}% 

%% for notes on emulateapj please see http://hea-www.harvard.edu/~alexey/emulateapj
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% End of code taken from emulateapj.cls %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% New as of august 2015:

%% These commands requested AAS
\newcommand{\noprint}[1]{}
\newcommand{\figsetstart}{{\bf Fig. Set} }
\newcommand{\figsetend}{}
\newcommand{\figsetgrpstart}{}
\newcommand{\figsetgrpend}{}
\newcommand{\figsetnum}[1]{{\bf #1.}}
\newcommand{\figsettitle}[1]{ {\bf #1}}
\newcommand{\figsetgrpnum}[1]{\noprint{#1}}
\newcommand{\figsetgrptitle}[1]{\noprint{#1}}
\newcommand{\figsetplot}[1]{\noprint{#1}}
\newcommand{\figsetgrpnote}[1]{\noprint{#1}}

\usepackage{url}
%% if we take away the xx before UrlBreaks we will get a url that breaks
%% at any letter or number. It might be better to break only at / however...
\expandafter\def\expandafter\xxUrlBreaks\expandafter{\UrlBreaks%  save the current one
  \do\a\do\b\do\c\do\d\do\e\do\f\do\g\do\h\do\i\do\j%
  \do\k\do\l\do\m\do\n\do\o\do\p\do\q\do\r\do\s\do\t%
  \do\u\do\v\do\w\do\x\do\y\do\z\do\A\do\B\do\C\do\D%
  \do\E\do\F\do\G\do\H\do\I\do\J\do\K\do\L\do\M\do\N%
  \do\O\do\P\do\Q\do\R\do\S\do\T\do\U\do\V\do\W\do\X%
  \do\Y\do\Z\do\1\do\2\do\3\do\4\do\5\do\6\do\7\do\8\do\9}


%% for tables continuing over pages
\usepackage{longtable}

%% for editing changes
\usepackage{xcolor}
% hyperref link defaults to "blue" (0000ff) as this matches our publisher produced pdf style
\definecolor{xlinkcolor}{cmyk}{1,1,0,0}


\PassOptionsToPackage{hyphens}{url}
%% In response to request from AAS
 \usepackage[bookmarks=true,         % show bookmarks bar?/ Changed March 22, 2019 for
                                     % improved accessibility
    pdfnewwindow=true,      % links in new window
   colorlinks=true,    % false: boxed links; true: colored links
  linkcolor=xlinkcolor,     % color of internal links
 citecolor=xlinkcolor,     % color of links to bibliography
filecolor=xlinkcolor,  % color of file links
urlcolor=xlinkcolor,      % color of external links
final=true,
 ]{hyperref}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Additions to  AASTeX by Amy Hendrickson, TeXnology Inc, August 17, 2015

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Table Tools, written by Amy Hendrickson
%% for American Astronomical Society
%% August 17, 2015
%% array package is necessary:


%%%%%%%%
%% 1) h in table preamble will makes text in that column ignored.

%%%%%%%%
%% 2) uppercase C L or R will make those columns in math mode.

%%%%%%%%
%% 3) \begin{splittabular}{<preamble B preamble>}...\end{splittabular}
%%    `B' in the preamble will show where the table should be broken. It may
%%    be used once in a preamble to break the table into two parts, or 
%%    two times in a preamble to break the table into three parts.

%%    The resulting table will position left half of table above right half of table.

%%    Care should be taken with \multicolumn so that it won't be broken between
%%    the two parts of the table. You may need to shorten the \multicolumn argument
%%    for number of columns to be appropriate to the number of columns in
%%    the new split table.

%%     In this example, for instance, we start with:
%%        \begin{tabular}{ccccccc}
%%        \multicolumn{7}{c}{\textsc{Table 1}}
%%     and change it to:
%%        \begin{splittabular}{ccccBccc}
%%        \multicolumn{4}{c}{\textsc{Table 1}}\\
%% 
%%     If you'd like to have a table number in the second half of the table,
%%     you can ask for a second multicolumn command that will position at the
%%     top of the second half of the table:
%%
%%        \begin{splittabular}{ccccBccc}
%%        \multicolumn{4}{c}{\textsc{Table 1}}&\multicolumn{3}{c}{\textsc{Table 1, Continued}}\\
%%
%%     You can do the same for the caption, and give a Continued caption for
%%     the second half of the table. Originally:
%%        \multicolumn{7}{c}{\textsc{Log of \textit{HST} Observations for
%%         NGC~6388}}
%%     Now, changed so that we get a caption on both halves of the table:
%%        \multicolumn{4}{c}{\textsc{Log of \textit{HST} Observations for
%%         NGC~6388}}&\multicolumn{3}{c}{\textsc{Log of \textit{HST} Observations for
%%         NGC~6388, Continued}}\\
%%     
%%     Similarly, you can rearrange table notes to appear at the bottom of
%%     the appropriate half of the table. For instance, starting with this,
%%     which would make table notes both appear at the bottom of the top half
%%     of the table:
%%        \multicolumn{4}{l}{\small{$\dagger$ Average distance of data set from
%%            cluster center.}}\\
%%        \multicolumn{4}{l}{$^{\rm a}$SNAP program.}\\
%%     
%%     We may substitute the following:
%%        \multicolumn{4}{l}{$^{\rm a}$SNAP program.}&\multicolumn{3}{l}{\small{$\dagger$ Average distance of data set from
%%            cluster center.}}\\
%%     
%%     Which will produce the first endnote underneath the top half of the
%%     table, and the second underneath the bottom half.
%%     
%%     Another option would be to have both endnotes appear below the bottom
%%     half of the table. Easily done:
%%        &&&&\multicolumn{3}{l}{$^{\rm a}$SNAP program.}\\
%%        &&&&\multicolumn{3}{l}{\small{$\dagger$ Average distance of data set from
%%            cluster center.}}\\
%%     %%     %%     %%     %%     %%     %%     

%%%%%%%%
%% 4) \colnumbers will make line with column numbers automatic. It will work with splittabular
%%     and splitdeluxetable as well as tabular.
%%     To use: type in \colnumbers within the table whereever you'd like it to appear, typically
%%     underneath the column headers, before the lines of data.

%%%%%%%%
%% 5) Easy Decimal numbering
%%    How to make decimal numbers in tables line up on the period:

%%     Use D (for decimal column) in table preamble for every decimal number. The decimal
%%     numbers will use two columns, one for the left part of the decimal
%%     number and one for the right part.

%%     D may be used more than once in a table preamble.

%%     If you want to type in a column header  over the decimal
%%     numbers, please use \multicolumn2c{} to span both columns.

%%     After the column headers, to start decimal numbering, 
%%     type in \decimals in the body of the table.

%%     When entering decimal numbers
%%     remember to leave a space after the decimal number, before the following &.
%%     For instance: &22.3 &35.96 \\

%%     If you'd like an empty entry, please supply a period and a space: & . &. You
%%     will not see the period in the resulting table.

%%     example:
%    \begin{tabular}{rDD}
%     \hline
%     &&&\multicolumn2c{\bf More}\\
%     &\multicolumn2c{\bf Decimals}&\multicolumn2c{\bf Decimals}\\
%     \hline
%     \hline
%     \decimals
%     one& . &34.2 \\
%     two &567.0 &21345 \\
%     three&.0 &62.5 \\
%     four&245 &5034.349923 \\
%     five&21 & \\
%     six& &21.6 
%    \end{tabular}
%%
%    Decimal numbering works within \begin{splittabular}...\end{splittabular}
%    \begin{splitdeluxetable} and \end{splitdeluxetable} for tables broken into two
%    or three parts. \colnumbers will number the columns counting both sides 
%    of the decimal number as one column; \tablehead{\colhead{}...} will make
%    the column headers position above the two columns used for one decimal
%    number as well.
%
%    Decimal numbers will be in math mode so that plus and minus signs are printed
%    correctly, expressions like `$\pm$ 1.2' or `\pm 1.2', will both work
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%











%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% end table tools
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Gridlines for positioning multiple illustrations in 
%% one figure environment.

\def\gridline#1{\vskip6pt\hbox to\hsize{#1}\vskip6pt}

\def\boxedfig#1#2#3{\hfill\fbox{\vbox{\parskip=0pt\relax
\hsize=#2
\includegraphics[width=#2]{#1}\vskip2pt\vtop{\hsize=#2
\centerline{#3}}}}\hfill}

\def\fig#1#2#3{\hfill\vbox{\parskip=0pt\hsize=#2
\includegraphics[width=#2]{#1}\vskip2pt\vtop{\centering
\footnotesize
\hsize=#2
#3\vskip1pt
}}\hfill}

\def\leftfig#1#2#3{\vbox{\parskip=0pt\relax\hsize=#2
\includegraphics[width=#2]{#1}\vskip2pt\vtop{\hsize=#2
\centering
#3\vskip1sp\vskip1sp}}\hfill}

\def\rightfig#1#2#3{\hfill\vbox{\parskip=0pt\relax\hsize=#2
\includegraphics[width=#2]{#1}\vskip2pt\vtop{\hsize=#2
\centering#3\vskip1sp}}}

\def\rotatefig#1#2#3#4{\hfill\vbox{\centering\parskip=0pt\hsize=#3
\includegraphics[width=#3,angle=#1]{#2}\vskip2pt\vtop{\centering
\footnotesize
\hsize=#3
#4\vskip1pt
}}\hfill}

%%%%%%%%%% End Grid line Macros %%%%%%%%%%

%%%%%%%%%% Color Editing Macros %%%%%%%%%%
%% \turnoffedit or \turnoffedits
%% will prevent all \edit<number>{text}, all
%% \collaborationcomment<number>{text}, and all
%% \authorcomment<number>{text} from producing any text.

\newif\ifturnoffedit
\def\turnoffedit{\global\turnoffedittrue}
\def\turnoffedits{\global\turnoffedittrue}
\let\turnoffediting\turnoffedits

\newif\ifturnoffone
\newif\ifturnofftwo
\newif\ifturnoffthree

%% \turnoffeditone to only turn off only \edit1, \collaborationcomment1 and \authorcomment1 :
\def\turnoffeditone{\turnoffonetrue}


%% \turnoffedittwo to only turn off only \edit2, \collaborationcomment2 and \authorcomment2 :
\def\turnoffedittwo{\turnofftwotrue}


%% \turnoffeditthree to only turn off only \edit3, \collaborationcomment3 and \authorcomment3 :
\def\turnoffeditthree{\turnoffthreetrue}


%% Other choices can be made, but this should be
%% standardized, so didn't make an user interface
%% to change the colors easily.

\expandafter\def\csname editcolor1\endcsname{black}% was magenta
\expandafter\def\csname editcolor2\endcsname{black}% was blue
\expandafter\def\csname editcolor3\endcsname{black}% was violet

\let\newgo\relax
\newcount\colorcount
	
\newcount\editnum
\def\edit#1#2{\ifcase#1\or\ifturnoffone%
\unskip%
\else%
\texorpdfstring{{\bf#2}}{#2}\fi\or%
\ifturnofftwo%
\unskip%
\else\texorpdfstring{{\bfseries\itshape#2}}{#2}\fi%
\or%
\ifturnoffthree\unskip\else%
\texorpdfstring{{\bfseries\underline{#2}}}{#2}\fi\fi}

\def\collaborationcomment#1#2{{\colorcount=#1
\ifturnoffedit\let\go\ignorespaces\else%
\let\go\relax%
\color{\csname editcolor\the\colorcount\endcsname}
\ifnum\colorcount=1{\ifturnoffone\else\bf (Collaboration~note: #2)\ \fi}\else
\ifnum\colorcount=2{\ifturnofftwo\else\it (Collaboration~note: #2)\ \fi}\else
\ifnum\colorcount=3{\ifturnoffthree\else (Collaboration~note: #2)\ \fi}\fi\fi\fi\fi%
\go}}

\def\authorcomment#1#2{{\colorcount=#1
\ifturnoffedit\let\go\ignorespaces\else%
\let\go\relax%
\color{\csname editcolor\the\colorcount\endcsname}
\ifnum\colorcount=1{\ifturnoffone\else\bf (Author~note 1: #2)\ \fi}\else
\ifnum\colorcount=2{\ifturnofftwo\else\it (Author~note 2: #2)\ \fi}\else
\ifnum\colorcount=3{\ifturnoffthree\else (Author~note 3: #2)\ \fi}\fi\fi\fi\fi%
\go}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Draft watermark

\definecolor{ltgray}{gray}{.9}% .89
\font\bighelv=phvr at 2in %
\def\setwatermarkfontsize#1{\font\bighelv=phvr at #1}

\def\watermark#1{\def\thewatermark{\hbox to\textwidth{\vtop to
1.1\textheight{\vss
\hskip24pt\rotatebox{60}{\hbox{\bighelv \color{ltgray} 
\uppercase{#1}}}\vss}}\hss}}
\let\thewatermark\empty

%% Used in titlepage definition, as \pagestyle{titlepage}
\def\ps@titlepage{%
  \let\@mkboth\@gobbletwo
\def\@oddhead{\ifx\thewatermark\empty\hfill\else
\hbox to \textwidth{\rlap{\thewatermark}\hfill}\fi}
\let\@oddfoot\@empty
\let\@evenhead\@empty\let\@evenfoot\@empty}

% end of watermark definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand*\submitjournal[1]
{\def\@submitted{Submitted to #1}}%

%% this one is used &&&
\def\titleblock@produce{%
 \begingroup
 \ltx@footnote@pop
 \def\@mpfn{mpfootnote}% 
 \def\thempfn{\thempfootnote}%
 \c@mpfootnote\z@
 \let\@makefnmark\frontmatter@makefnmark
  \frontmatter@setup
  \thispagestyle{titlepage}\label{FirstPage}%
\ifmodern\leftskip=0pt\rightskip\leftskip\fi
  \frontmatter@title@produce
  \groupauthors@sw{%
\frontmatter@author@produce@group
  }{%
   \frontmatter@author@produce@script
  }%
  \frontmatter@RRAPformat{%
\vskip12pt
   \expandafter\produce@RRAP\expandafter{\@date}%
   \expandafter\produce@RRAP\expandafter{\@received}%
   \expandafter\produce@RRAP\expandafter{\@revised}%
   \expandafter\produce@RRAP\expandafter{\@accepted}%
   \expandafter\produce@RRAP\expandafter{\@published}%
  }%
\expandafter\ifx\csname @submitted\endcsname\relax\else
\vskip6pt
\expandafter\produce@RRAP\expandafter{\centerline{\@submitted\hbox
to 20pt{\hfill}}\vskip12pt}%
\fi
  \frontmatter@abstract@produce
  \@ifx@empty\@pacs{}{%
   \@pacs@produce\@pacs
  }%
  \@ifx@empty\@keywords{}{%
   \@keywords@produce\@keywords
  }%
  \par
  \frontmatter@finalspace
\endgroup%
}%


% needed??
\gdef\specialbibitem#1[#2]#3#4{
{\tt\string\bibitem}[#2]{\tt\string{\string}}
#4}



%%%%%%%%%%%%%%%%%%%%%%%%%
%% Track Changes
%% Amy Hendrickson, Nov 2015
%% Change Jan 2016, to allow list of changes
%% to give line and page numbers for more than one entry on the same line.
%% Change Feb 2016, to allow optional argument for time/date, and/or editor initials, etc.


\providecolor{trackchange}{cmyk}{0,0,0,1}
\providecolor{explain}{cmyk}{0,0,0,1}

\newif\ifsilent

\newcount\refchangenumber
\def\added{\@ifnextchar[{\xadded}{\yadded}}

\long\def\xadded[#1]#2{%
\iftrack {\global\advance\refchangenumber by 1\relax%
\vtop to 0pt{\vss
\hypertarget{link\the\refchangenumber}{}
\vskip14pt}%
\ifnumlines%
\ifabstract\else%
\xdef\doit{\noexpand\linelabel{\the\refchangenumber}}\doit\fi\else%
\xdef\doit{\noexpand\label{\the\refchangenumber}{}{}{}}\doit\fi}%
{\color{trackchange}\bf(Added: [#1] #2)}%%
\ifabstract\label{\the\refchangenumber}%
\expandafter\gdef\csname
changenum\the\refchangenumber\endcsname{\bf Added: [#1]
\textcolor{trackchange}\bf\relax{#2}\global\silenttrue}%
\else\expandafter\gdef\csname
changenum\the\refchangenumber\endcsname{\bf\relax Added: [#1]
\textcolor{trackchange}\bf\relax{\let\bibitem\specialbibitem #2}\global\silentfalse}\fi%
\else#2\fi}



\long\def\yadded#1{%
\iftrack{\global\advance\refchangenumber by 1\relax%
\vtop to 0pt{\vss
\hypertarget{link\the\refchangenumber}{}
\vskip14pt}%
\ifnumlines\ifabstract\else%
\xdef\doit{\noexpand\linelabel{\the\refchangenumber}}\doit\fi\else%
\xdef\doit{\noexpand\label{\the\refchangenumber}{}{}{}}\doit%
\fi}%
{\color{trackchange}\bf(Added: #1)}%%
\ifabstract%
\label{\the\refchangenumber}%
\expandafter\gdef\csname
changenum\the\refchangenumber\endcsname{\bf Added:
\textcolor{trackchange}\bf\relax{#1},
\global\silenttrue}\else
\expandafter\gdef\csname
changenum\the\refchangenumber\endcsname{\bf Added:
\textcolor{trackchange}{\bf\relax\let\bibitem\specialbibitem #1},%
\global\silentfalse}\fi%
\else#1\fi}



\def\deleted{\@ifnextchar[{\xdeleted}{\ydeleted}}

\long\def\xdeleted[#1]#2{\iftrack{\global\advance\refchangenumber by
1\relax%
\vtop to 0pt{\vss
\hypertarget{link\the\refchangenumber}{}
\vskip14pt}%
\ifnumlines\ifabstract\else%
\xdef\doit{\noexpand\linelabel{\the\refchangenumber}}\doit\fi%
\else%
\xdef\doit{\noexpand\label{\the\refchangenumber}}\doit\fi%
}%
{\color{trackchange}\bf%
\ifbib\let\sout\relax\fi%
(Deleted: [#1] \sout{#2})
}%
\ifabstract\label{\the\refchangenumber}%
\expandafter\gdef\csname
changenum\the\refchangenumber\endcsname{Deleted: [#1]
{\color{trackchange}\bf%
\sout{#2}}\global\silenttrue}%
\else
\expandafter\gdef\csname
changenum\the\refchangenumber\endcsname{Deleted: [#1] 
{\color{trackchange}%
\let\bibitem\specialbibitem%
\sout{#2}
}\global\silentfalse}\fi%
\fi}

\long\def\ydeleted#1{\iftrack{\global\advance\refchangenumber by 1\relax%
\vtop to 0pt{\vss
\hypertarget{link\the\refchangenumber}{}
\vskip14pt}%
\ifnumlines\ifabstract\else%
\xdef\doit{\noexpand\linelabel{\the\refchangenumber}}\doit\fi%
\else%
\xdef\doit{\noexpand\label{\the\refchangenumber}{}{}{}}\doit%
\fi}%
%%
{\color{trackchange}\bf%
\ifbib\let\sout\relax\fi%
(Deleted: \sout{#1})}%
\ifabstract\label{\the\refchangenumber}%
\expandafter\gdef\csname changenum\the\refchangenumber\endcsname{Deleted:
{\color{trackchange}\bf
\sout{#1}}\global\silenttrue}%
\else
\expandafter\gdef\csname changenum\the\refchangenumber\endcsname{Deleted:
{\color{trackchange}\bf%
\sout{#1}}\global\silentfalse}\fi%
\fi}


\def\replaced{\@ifnextchar[{\xreplaced}{\yreplaced}}

\long\def\xreplaced[#1]#2#3{%
\iftrack\global\advance\refchangenumber by 1\relax%
\vtop to 0pt{\vss
\hypertarget{link\the\refchangenumber}{}
\vskip14pt}%
\ifnumlines\ifabstract\else\xdef\doit{\noexpand\linelabel{\the\refchangenumber}}\doit\fi\else%
\xdef\doit{\noexpand\label{\the\refchangenumber}}\doit\fi%
{\ifbib\let\sout\relax\fi
\color{trackchange}\bf(Replaced: [#1] \sout{#2}}%
{\color{black}replaced with:} {
\color{trackchange}\bf\relax #3)}%
\ifabstract\label{\the\refchangenumber}%
\expandafter\gdef\csname
changenum\the\refchangenumber\endcsname{Replaced: [#1]
{\color{trackchange}\bf\relax\sout{#2}} {\color{black} replaced with:}
{\color{trackchange}\bf\relax#3}, \global\silenttrue}%
\else
\expandafter\gdef\csname
changenum\the\refchangenumber\endcsname{Replaced: [#1]
{\ifbib\let\sout\relax\fi
\color{trackchange}\bf\relax\sout{#2}
} {\color{black} replaced with:}
{\color{trackchange}\bf\relax#3}, \global\silentfalse}\fi%
\else#3\fi}

\long\def\yreplaced#1#2{%
\iftrack\global\advance\refchangenumber by 1\relax%
\vtop to 0pt{\vss
\hypertarget{link\the\refchangenumber}{}
\vskip14pt}%
\ifnumlines\ifabstract\else\xdef\doit{\noexpand\linelabel{\the\refchangenumber}}\doit\fi\else%
\xdef\doit{\noexpand\label{\the\refchangenumber}}\doit\fi%
{\ifbib\let\sout\relax\fi
\color{trackchange}\bf\relax(Replaced: 
\sout{#1}
}%
{\color{black}replaced with:}
{\color{trackchange}\bf\relax #2)}%
\ifabstract\label{\the\refchangenumber}%
\expandafter\gdef\csname changenum\the\refchangenumber\endcsname{Replaced:
{\color{trackchange}\bf\relax
\sout{#1}} {\color{black} replaced with:}
{\color{trackchange}\bf\relax#2},\global\silenttrue}%
\else
\expandafter\gdef\csname
changenum\the\refchangenumber\endcsname{Replaced:
{\ifbib\let\sout\relax\fi
\color{trackchange}\bf\relax\sout{#1}
} {\color{black} replaced with:}
{\let\bibitem\specialbibitem\color{trackchange}\bf\relax#2}, \global\silentfalse}\fi%
\else#2\fi}

\def\explain{\@ifnextchar[{\xexplain}{\yexplain}}

\def\xexplain[#1]#2{\iftrack\ {\bfseries\itshape\color{explain} [Explanation of change:
#2 (#1)]\ }\fi}

\def\yexplain#1{\iftrack\ {\bfseries\itshape\color{explain} [Explanation of change:
#1]\ }\fi}


\newcount\listchangenum

\def\listofchanges{
\clearpage
\iftrack
\ifnum\refchangenumber>0
\ifnumlines\nolinenumbers\fi
\vskip36pt
\hrule
\noindent{\vrule height 14pt width0pt depth 6pt\large\bf List of Changes}
\hrule
\vskip18pt
\nobreak
{\parskip=4pt \parindent=0pt
\loop
\global\silentfalse
\vskip-1pt\relax
\global\advance\listchangenum by 1\relax
\expandafter\ifx\csname changenum\the\listchangenum\endcsname\relax
\else
\csname changenum\the\listchangenum\endcsname\ 
on page
%% can't get hyperlink to work correctly for page numbers, works but error messages.
\hyperlink{link\the\listchangenum}{\bf\pageref{\the\listchangenum}}%
\ifnumlines%
\ifsilent\global\silentfalse \else%
\setbox0=\hbox{\lineref{\the\listchangenum}}%
\ifdim\wd0<20pt%
, line\
\hyperlink{link\the\listchangenum}{\lineref{\the\listchangenum}}%
\fi
\fi\fi.\relax
\repeat}
\fi\fi
\thispagestyle{empty}
}


\PassOptionsToPackage{normalem}{ulem}
\usepackage{ulem}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% From earlier version of AASTeX, for usefulness and 
%% backward compatibility, with some requested additions

\let\jnl@style=\rmfamily
\def\ref@jnl#1{{\jnl@style#1}}%

%%%
\renewcommand\memsai{\ref@jnl{Mem.~Soc.~Astron.~Italiana}}%
          % Mem. Societa Astronomica Italiana
%% longer version:
\newcommand\memsocai{Mem. Societ\`a Astronomica Italiana}
\newcommand\aspconf{Ast. Soc. of the Pac. Conference Series}

\let\astap=\aap
\let\apjlett=\apjl
\let\apjsupp=\apjs
\let\applopt=\ao

%%% More useful commands from Earlier version of Aastex:
\let\la=\lesssim            % For Springer A&A compliance...
\let\ga=\gtrsim

\let\farcm\farcm@mss
\let\farcs\farcs@mss

\def\farcm@apj{%
 \mbox{.\kern -0.7ex\raisebox{.9ex}{\scriptsize$\prime$}}%
}%

\def\farcs@apj{%
 \mbox{%
  \kern  0.13ex.%
  \kern -0.95ex\raisebox{.9ex}{\scriptsize$\prime\prime$}%
  \kern -0.1ex%
 }%
}%

\def\chem@bnd#1{%
 {%
  \kern.1em\relax
  \setbox\z@\hbox{M}%
  \dimen@ii.8em\relax
  \p@=.1em\relax
  \dimen@.5\ht\z@\dimen@i-\dimen@
  \advance\dimen@1.5\p@\advance\dimen@i-1.0\p@
  #1%
  \kern.1em\relax
  }%
 }%
\def\@sbnd{%
 \advance\dimen@-1.5\p@\advance\dimen@i1.5\p@
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 }%
\def\@dbnd{%
 \advance\dimen@-0.5\p@\advance\dimen@i0.5\p@
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 \advance\dimen@-1.5\p@\advance\dimen@i1.5\p@
 \hskip-\dimen@ii
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 }%
\def\@tbnd{%
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 \advance\dimen@-1.5\p@\advance\dimen@i1.5\p@
 \hskip-\dimen@ii
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 \advance\dimen@-1.5\p@\advance\dimen@i1.5\p@
 \hskip-\dimen@ii
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 }%

\renewcommand\LaTeX{%
 \leavevmode
 L%
 \raise.42ex\hbox{%
  \count@=\the\fam
  $\fam\count@\scriptstyle\kern-.3em A$%
 }%
 \kern-.15em\TeX
}%

%% added Feb 2, 2016, redefining definition of \caption made with hyperref
%% package. Making \caption[] be the default, to prevent sending caption 
%% text to listoftables or listoffigures, which we are not going to use
%% anyway. This change enables track changes commands to work in captions.
\def\xtable{table}
\def\caption{\numlinesfalse
\ifx\@captype\@undefined 
\@latex@error {\noexpand \caption outside float}\@ehd 
\expandafter \@gobble \else 
\H@refstepcounter \@captype \let \Hy@tempa \@caption \@ifundefined
{float@caption}{}
{\expandafter \ifx \csname @float@c@\@captype\endcsname
\float@caption\let\Hy@tempa\Hy@float@caption\fi} 
\expandafter\@firstofone\fi 
{\@dblarg {\Hy@tempa \@captype}}[]}

\def\@caption#1[#2]#3{{\small\rm\expandafter \ifx \csname if@capstart\expandafter \endcsname 
\csname iftrue\endcsname \global \let \@currentHref \hc@currentHref \else \hyper@makecurrent {\@captype }\fi 
\@ifundefined {NR@gettitle}{\def \@currentlabelname{\thesection\relax #2}}
{\NR@gettitle {#2}}\par 
%\addcontentsline {\csname ext@#1\endcsname }{#1}{\protect \numberline {\csname the#1\endcsname }{\ignorespaces #2}}
\begingroup \@parboxrestore \if@minipage \@setminipage \fi \normalsize 
\expandafter\ifx \csname if@capstart\expandafter \endcsname 
\csname iftrue\endcsname \global \@capstartfalse 
\@makecaption {\bf\csname fnum@#1\endcsname }{\ignorespaces #3}
\else \@makecaption {\bf\csname fnum@#1\endcsname }{\ignorespaces \ifHy@nesting 
\expandafter \hyper@@anchor \expandafter {\@currentHref }{#3}\else 
\Hy@raisedlink {\expandafter \hyper@@anchor \expandafter
{\@currentHref }{\relax }}{#3}\fi }\fi \par
\endgroup}}

%% changed, March 23, 2019, took out \\ at end of definition:
\newcommand\tablebreak{\\[-11pt]\noalign{\break}}

%% As suggested by Greg Schwarz, August Meunch, Feb 11

\def\dataset{\def\doi##1{https://doi.org/##1}
\@ifnextchar[{\ydataset}{\xdataset}}
\def\xdataset#1{\ydataset[]{#1}\let\doi\savedoi}
\def\ydataset[#1]#2{\def\one{#1}\ifx\one\empty
\href{#2}{[DATASET]}\else
\href{#2}{#1}\fi\let\doi\savedoi}

%% this seems in the middle of nowhere...
\advance\columnsep6pt

%%%%%%%%%%
%% macros to give hyperref link to authors when orcid numbers
%% are supplied with \author[0000-2345-3333-0023]{author name}
 \def\lookforbracket{\ifx\next[\let\go\xauthor\else\let\go\yauthor\fi\go}

\newcount\entriesinthiscollab 
\newcount\allentries
\newif\ifseesmessage
\def\author{\global\advance\entriesinthiscollab by 1\relax%
\global\advance\allentries by 1\relax\futurelet\next\lookforbracket}

\def\new@author@def#1#2{%
 \move@AU\move@AF\move@AUAF
 \let\@AF@join\@author@join
 \def\@author{{\href{http://orcid.org/#1}{#2% 
\openin1 Orcid-ID.png \ifeof1
%% message for authors??
%\typeout{^^J^^J  ! Missing File: Orcid-ID.png; needed for Orcid Author icon !
%^^J}
\else%
\hskip2pt\includegraphics[width=9pt]{Orcid-ID.png}\fi}}{}}%
}%

\def\orciderrormessage{%
\typeout{^^J^^J [\firstarg]\space Invalid ORCID Identifier!^^J^^J The ID
should consist of four sets of four digits,^^J separated with -, ie,
0000-0012-3245-1234 or ^^J
0000-0012-3245-123X
^^J^^J
Please fix, or don't use square bracket argument with
\string\author[]\string{\string}
^^J^^J}%
}

%% Tests to see if author gave reasonable Orcid number;
%% if not, stops processing and gives error message.
\newif\ifbadOrcid
\def\lookatOrcid #1-#2-#3-#4-{%
\def\one{#1}\def\two{#2}\def\three{#3}\def\four{#4}%
\ifx\one\empty%
\global\badOrcidtrue%
\else%
\ifx\two\empty%
\global\badOrcidtrue%
\else%
\ifx\three\empty%
\global\badOrcidtrue%
\else%
\ifx\four\empty%
\global\badOrcidtrue%
\fi\fi\fi\fi}

\def\lookforDigits #1#2#3#4{
\def\xone{#1}\def\xtwo{#2}\def\xthree{#3}\def\xfour{#4}
%%
\ifx\xone\empty 
\global\badOrcidtrue
\else
  \ifcat \xone 1\else
  \global\badOrcidtrue\fi
\fi
%%%
\ifx\xtwo\empty 
\global\badOrcidtrue
\else
 \ifcat \xtwo 1\else
  \global\badOrcidtrue\fi
\fi
%%%
\ifx\xthree\empty 
\global\badOrcidtrue
\else
 \ifcat \xthree 1\else
  \global\badOrcidtrue\fi
\fi
%%%
\ifx\xfour\empty 
\global\badOrcidtrue
\else
 \ifcat \xfour 1\else
  \global\badOrcidtrue\fi
\fi}

\def\FinallookforDigits #1#2#3#4{
\def\xone{#1}\def\xtwo{#2}\def\xthree{#3}\def\xfour{#4}
%%
\ifx\xone\empty 
\global\badOrcidtrue
\else
  \ifcat \xone 1\else
  \global\badOrcidtrue\fi
\fi
%%%
\ifx\xtwo\empty 
\global\badOrcidtrue
\else
 \ifcat \xtwo 1\else
  \global\badOrcidtrue\fi
\fi
%%%
\ifx\xthree\empty 
\global\badOrcidtrue
\else
 \ifcat \xthree 1\else
  \global\badOrcidtrue\fi
\fi
%%%
\ifx\xfour\empty 
\global\badOrcidtrue
\else
   \ifcat \xfour 1\else
      \if\xfour X
      \else
      \global\badOrcidtrue\fi%
   \fi%
\fi}





%% Last digit might be X
\def\xauthor[#1]#2{\def\testone{#1-}\def\firstarg{#1}%
\setbox0=\hbox{\expandafter\lookatOrcid\testone {}-{}-{}-{}-
\expandafter\lookforDigits\one {}{}{}{}%
\expandafter\lookforDigits\two {}{}{}{}%
\expandafter\lookforDigits\three {}{}{}{}%
\expandafter\FinallookforDigits\four {}{}{}{}}%
%%
\ifbadOrcid\let\go\orciderrormessage
\else\let\go\relax\fi\go%
\ifbadOrcid\let\go\stop\else\let\go\relax\fi\go%
\new@author@def{#1}{#2}}%% oct 7, this doesn't seem to work

\def\yauthor{\@author@def{}}

\def\altaffilmark#1{\typeout{^^J^^J\string\altaffilmark\space is no longer
needed. It will not do anything.^^J Please use
\string\altaffiliation\string{\string} instead.^^J^^J}\stop}

\def\altaffiltext#1{\typeout{^^J^^J\string\altaffiltext\space is no longer
needed. It will not do anything.^^J Please use
\string\altaffiliation\string{\string} instead.^^J^^J}\stop}

%% We do not need \and in list of authors. It is taken care of
%% automatically. Below is the definition from revtex4-1:
% 
%\renewcommand\frontmatter@and{\class@err{\protect\and\space is not
% supported}}

\newcount\AuthorCollaborationLimit
\let\AuthorCallLimit\AuthorCollaborationLimit
%% no limit for default
\AuthorCollaborationLimit=10000
\newcount\largestAffilNum

\def\lookfornumbers#1#2#3#4#5#6#7#8#9{\def\one{#1}
\def\two{#2}
\def\three{#3}
\def\four{#4}
\def\five{#5}
\def\six{#6}
\def\seven{#7}
\def\eight{#8}
\def\nine{#9}
\ifnum\one>\largestAffilNum
\global\largestAffilNum=\one\fi
%%%
\ifx\two\empty\else
\ifnum\two>\largestAffilNum
\global\largestAffilNum=\two\fi\fi
%%%
\ifx\three\empty\else
\ifnum\three>\largestAffilNum
\global\largestAffilNum=\three\fi\fi
%%%
\ifx\four\empty\else
\ifnum\four>\largestAffilNum
\global\largestAffilNum=\four\fi\fi
%%%
\ifx\five\empty\else
\ifnum\five>\largestAffilNum
\global\largestAffilNum=\five\fi\fi
%%%
\ifx\six\empty\else
\ifnum\six>\largestAffilNum
\global\largestAffilNum=\six\fi\fi
%%%
\ifx\seven\empty\else
\ifnum\seven>\largestAffilNum
\global\largestAffilNum=\seven\fi\fi
%%%
\ifx\eight\empty\else
\ifnum\eight>\largestAffilNum
\global\largestAffilNum=\eight\fi\fi
%%%
\ifx\nine\empty\else
\ifnum\nine>\largestAffilNum
\global\largestAffilNum=\nine\fi\fi
}

%\gdef\newcomma@space{\hskip-3pt\textsuperscript{,}}%
\def\doEtAl{\rm et al.\gdef\doEtAl{\relax}}
%% =====================
\newif\iffirsttime
\firsttimetrue
\newcount\totalentries
\newcount\docollabnum
\newcount\tempauthornumber
\newcount\countauthors
\newif\ifdothis
\def\doAnd{}
\newcount\testnum

\def\olddoauthor#1#2#3{%
\iflongauthor\vskip6pt\fi
\global\advance\countauthors by 1
\ifnum\countauthors>\AuthorCollaborationLimit
\doEtAl
\else
\ifnum\AuthorCollaborationLimit=1
\let\@listand\relax
\fi
  \ignorespaces\leavevmode\hbox{#1\unskip\@listcomma}% nice, keeps name from breaking across lines
\fi
  \begingroup
\ifnum\countauthors>\AuthorCollaborationLimit\else
  #3% all affil numbers
\ifx\@affilID@temp\empty
\else%
\setbox0=\hbox{\expandafter\lookfornumbers\@affilID@temp{}{}{}{}{}{}{}{}{}}%
\fi\fi
%% #2= \altaffiliation{} or \email{} or 
%% possibly anything other than author, affiliation, or collaboration 
  \@if@empty{#2}{\endgroup{}{}}
{\ifnum\countauthors>\AuthorCollaborationLimit\endgroup{}{}%% <<< bug fix, added \endgroup{}{}
\else
\endgroup{\comma@space}{}\frontmatter@footnote{#2}\fi}%
\ifnum\countauthors>\AuthorCollaborationLimit\else  \space
\@listand\fi 
}%

\newif\ifaddspace
\def\doauthor#1#2#3{%
\iflongauthor\vskip6pt\fi%%%
\ifanonymous%
    \iffirsttime%
     \global\firsttimefalse%
     Anonymous author(s)%
    \fi%
\else%% ends at end of this def
%%%
\ifnum\docollabnum< 1
\global\AuthorCollaborationLimit\expandafter\csname
currCollabLimit0\endcsname%
%% this won't change until after collaboration name at end
\fi %% end ifnum\docollabnum
%%
%%%%%%%%%%%%%%%%
\global\advance\totalentries by 1\relax%
\global\advance\countauthors by 1\relax%
%%%
\ifallauthors\global\AuthorCollaborationLimit=9999 \fi%
%%%
% for testing
% [author number=\the\countauthors/ auth collab limit
% =\the\AuthorCollaborationLimit]
%
\gdef\xone{#1}%
\ifnum\countauthors < \AuthorCollaborationLimit
\global\addspacetrue
\gdef\docomma{,}\else\gdef\docomma{}\fi% ok
%%%
\ifnum\countauthors = \AuthorCollaborationLimit
\ifnum\AuthorCollaborationLimit=1\else
\gdef\doAnd{And }\fi%
\ifnum\tempauthornumber= 1
\gdef\doAnd{  } \fi\fi%
%%
{\tempauthorminusone=\AuthorCollaborationLimit
\advance\tempauthorminusone by -1
\ifnum\countauthors=\tempauthorminusone
\gdef\doAnd{And }%
\gdef\docomma{}\fi}%
%%%
\global\dothisfalse%
%% if num countauthors is less than or equal to \AuthorCollaborationLimit, print name
\ifnum\countauthors< \AuthorCollaborationLimit
%%% 
\ifx\xone\empty\else%
\global\dothistrue%
%\expandafter\gdef\csname
%dothisaffil-\the\countauthors\the\docollab\endcsname{dothisone}
  \ignorespaces\leavevmode\hbox{#1\unskip\docomma}% nice, keeps name from breaking across lines
\fi%
\fi%
%%
\ifnum\countauthors= \AuthorCollaborationLimit
%%% 
\ifx\xone\empty\else
\global\dothistrue
  \ignorespaces\leavevmode\hbox{\doAnd #1\unskip\docomma}%nice, keeps name from breaking across lines
\fi%
\fi%
%% ++++
%%
\ifsuppressAffiliations\else%
\ifx\xone\empty\else%
\begingroup%
\ifnum\countauthors>\AuthorCollaborationLimit\else
  #3% all affil numbers
\ifx\@affilID@temp\empty %% number following author
\else%
\setbox0=\hbox{\expandafter\lookfornumbers\@affilID@temp{}{}{}{}{}{}{}{}{}}%
\fi\fi%
%% #2= \altaffiliation{} or \email{} or 
%% possibly anything other than author, affiliation, or collaboration 
  \@if@empty{#2}{\endgroup{}{}}%
{\ifnum\countauthors>\AuthorCollaborationLimit\endgroup{}{}%% <<< bug fix, added \endgroup{}{}
\else
\endgroup{\comma@space}{}\frontmatter@footnote{#2}\fi}%
\fi%% end test of empty
\fi%% end test of suppressAffiliations
%%%%%%%%%%%%%%%%%%%%%%%%
%%% Collaboration name is always used if available:
\expandafter\ifx\csname currCollabName\the\totalentries\endcsname\relax\else 
%%%
%%%
%%%
%% changed locally:
{\advance\docollabnum 1 
\advance\countauthors-1
%%%
%%% We need to test to see if more than one author in collaboration
%%% and if the number of authors is greater than the \AuthorCollaborationLimit.
%%% If both of these are true, than use `et al.' , otherwise do not:
\expandafter\ifx\csname
CollabTotalAuthors\the\docollabnum\endcsname\relax
\else
\expandafter\ifnum\csname
CollabTotalAuthors\the\docollabnum\endcsname
< 2 %%%%% Don't use et al if there is only one author
\else
\expandafter\ifnum\csname CollabTotalAuthors\the\docollabnum\endcsname
>\AuthorCollaborationLimit\relax%
\ifnum\AuthorCollaborationLimit=0\else
\vskip3pt
{\rm et al.} \vskip-3pt
\fi\fi\fi\fi%
}%% end local change to docollabnum
%%%%
\expandafter\ifx\csname
currCollabName\the\totalentries\endcsname\empty
\vskip-3pt
\else
\vskip6pt
\expandafter\csname currCollabName\the\totalentries\endcsname\vskip8pt
\affiliation{testing}
\fi%
%%%%%%%%
\global\countauthors=0
%%%
%%%
\global\advance\docollabnum  by 1
%% Set up counters for next time through this loop:
{\advance\docollabnum by1
  \expandafter\ifx\csname
  specificCollabLimit\the\docollabnum\endcsname\relax
 \else
   \global\AuthorCollaborationLimit=\csname
   specificCollabLimit\the\docollabnum\endcsname 
  \fi%
\expandafter\ifx\csname
CollabTotalAuthors\the\docollabnum\endcsname\relax
\else
\global\tempauthornumber=\csname
CollabTotalAuthors\the\docollabnum\endcsname
\fi}%
\fi %% ends test to see if it is time to use collaboration name
\fi %% ends ifanonymous
\ifaddspace\ \fi\global\addspacefalse}% oct 7, 2020, allows multiple names to break over lines,
                                         %but no extra space if \collaboration{1}


%%%% ++++====
\newcount\tempauthorminusone
%% +++
\def\doAllauthors#1#2#3{%
\global\suppressAffiliationsfalse
\iflongauthor\vskip6pt\fi
%%%
\ifanonymous
    \iffirsttime
     \global\firsttimefalse
     Anonymous author(s)
    \fi
\else %% ends at end of this def
%%%
\ifnum\docollabnum< 1
%% these won't change until after collaboration name at end
\global\AuthorCollaborationLimit\expandafter\csname currCollabLimit0\endcsname 
\global\tempauthornumber=\csname CollabTotalAuthors1\endcsname
\fi%% end ifnum\docollabnum
%%
%%%%%%%%%%%%%%%%
\global\advance\totalentries by 1
\global\advance\countauthors by 1
%%%
\ifallauthors\global\AuthorCollaborationLimit=9999 \fi
%%%
% for testing
%[author number=\the\countauthors/ temp author
%=\the\tempauthornumber]
\def\one{#1}
{\tempauthorminusone=\tempauthornumber
\advance\tempauthorminusone by -1
\ifnum\countauthors < \tempauthornumber
\gdef\xdocomma{,}%
\else\gdef\xdocomma{}\fi%
%%%
\ifnum\countauthors = \tempauthornumber
\gdef\xdoAnd{And}\gdef\xdocomma{}%
\else\gdef\xdoAnd{}\fi%
\ifnum\tempauthornumber= 1
\gdef\xdoAnd{}\gdef\xdocomma{}\fi%
\ifnum\countauthors=\tempauthorminusone
\gdef\xdocomma{}\fi%
}
%%%
%% if num countauthors is less than or equal to \AuthorCollaborationLimit, print name
\ifnum\countauthors< \AuthorCollaborationLimit
%%% 
\ifx\one\empty\else
  \ignorespaces\leavevmode\hbox{\unskip\xdoAnd\ #1\unskip\xdocomma}% nice, keeps name from breaking across lines
\fi\fi%
%%
\ifnum\countauthors= \AuthorCollaborationLimit
%%% 
\ifx\one\empty\else
\ignorespaces\leavevmode\hbox{\unskip\xdoAnd\ #1\unskip\xdocomma}% nice, keeps name from breaking across lines
\fi%
\fi%
%%
\ifx\one\empty\else
 \begingroup
\ifnum\countauthors>\AuthorCollaborationLimit\else
  #3% all affil numbers
\ifx\@affilID@temp\empty
\else%
\setbox0=\hbox{\expandafter\lookfornumbers\@affilID@temp{}{}{}{}{}{}{}{}{}}%
\fi\fi
%% #2= \altaffiliation{} or \email{} or 
%% possibly anything other than author, affiliation, or collaboration 
  \@if@empty{#2}{\endgroup{}{}}
{\ifnum\countauthors>\AuthorCollaborationLimit\endgroup{}{}%% <<< bug fix, added \endgroup{}{}
\else
\endgroup{\comma@space}{}\frontmatter@footnote{#2}\fi}%
\fi%% end test of empty
%%%%%%%%%%%%%%%%%%%%%%%%
%%% Collaboration name is always used if available:
\expandafter\ifx\csname currCollabName\the\totalentries\endcsname\relax\else 
%%%
%% No et al in allauthors, because all authors are listed!
%%
\expandafter\ifx\csname
currCollabName\the\totalentries\endcsname\empty
\else
\vskip6pt
\expandafter\csname currCollabName\the\totalentries\endcsname\vskip6pt
\fi
\global\countauthors=0
%%%
%%%
\global\advance\docollabnum  by 1
%% Set up counters for next time through this loop:
{\advance\docollabnum by1
  \expandafter\ifx\csname
  specificCollabLimit\the\docollabnum\endcsname\relax
 \else
   \global\AuthorCollaborationLimit=\csname
   specificCollabLimit\the\docollabnum\endcsname 
  \fi
%
\expandafter\ifx\csname
CollabTotalAuthors\the\docollabnum\endcsname\relax
\else
\global\tempauthornumber=\csname
CollabTotalAuthors\the\docollabnum\endcsname
\fi}
\fi %% ends test to see if it is time to use collaboration name
\fi %% ends ifanonymous
}% 

%%%
%% set \affil to match \affiliation found in revtex, since authors are accustomed to using \affil{}
\let\affil\affiliation

\newif\iffirstaffil
\firstaffiltrue

\newcount\affilnum
%% +++!
\def\@affil@script#1#2#3#4{%
\ifsuppressAffiliations\else
\iffirstaffil
% Oct 2017
\vskip2pt
\global\firstaffilfalse\fi
 \@ifnum{#1=\z@}{}{%
  \par
  \begingroup
   \frontmatter@affiliationfont
   \@ifnum{\c@affil<\affil@cutoff}{}{%
\def\one{#1}
%%
%%
\ifnum\one<\largestAffilNum
%% this makes the numbers
   \def\@thefnmark{#1}\@makefnmark\fi
\ifnum\one=\largestAffilNum
   \def\@thefnmark{#1}\@makefnmark\fi
   }%
\ifnum\one<\largestAffilNum
   \ignorespaces#3%
\fi
\ifnum\one=\largestAffilNum
   \ignorespaces#3%
\fi
   \@if@empty{#4}{}{\frontmatter@footnote{#4}}%
   \par
  \endgroup
 }%
\fi}%



\newif\ifnobreakafterkeywords
\def\NoNewPageAfterKeywords{\global\nobreakafterkeywordstrue}
\NoNewPageAfterKeywords
\def\NewPageAfterKeywords{\global\nobreakafterkeywordsfalse}

%%%+++!!!
\def\frontmatter@maketitle{%
%% \@author@finish needs to be here:
\@author@finish
\if@firstsection% not appendix, in other words
  \title@column\titleblock@produce 
\onecolumngrid
\else
%
\global\firstaffiltrue
 \title@column\secondtitleblock@produce
\fi
  \suppressfloats[t]%
%% Prevent these from being turned off so that
%% we can use \maketitle again for \AllAuthors.
%%
%  \let\and\relax
  \let\affiliation\@gobble
  \let\author\@gobble
%  \let\@AAC@list\@empty
%  \let\@AFF@list\@empty
%  \let\@AFG@list\@empty
%  \let\@AF@join\@AF@join@error
%  \let\email\@gobble
%  \let\@address\@empty
% \let\maketitle\relax
%  \let\thanks\@gobble
\if@firstsection
\let\abstract\@undefined\let\endabstract\@undefined
\titlepage@sw{%
\ifnobreakafterkeywords
\else
\vfil
\ifrnaas\else \clearpage\fi
\fi
 }{}%
\fi
}%


\let\maketitle\frontmatter@maketitle

\def\secondtitleblock@produce{%
 \begingroup
  \ltx@footnote@pop
  \def\@mpfn{footnote}%
  \def\thempfn{\thefootnote}%
  \c@footnote\z@
  \let\@makefnmark\frontmatter@makefnmark
%  \frontmatter@setup
  \thispagestyle{titlepage}%\label{FirstPage}%
%  \frontmatter@title@produce
  \groupauthors@sw{%
   \frontmatter@author@produce@group
  }{%
   \frontmatter@author@produce@script
  }%
  \endgroup
}%

%% no club or widow lines
\widowpenalty=10000
\clubpenalty\widowpenalty
\setlength{\footnotesep}{8pt}

\ifmodern
\setlength{\voffset}{0in}
\setlength{\hoffset}{0in}
\setlength{\textwidth}{6in}
\setlength{\textheight}{9.2in}
\setlength{\headheight}{0ex}
\setlength{\headsep}{36pt} % this is 2 lines in ``manuscript''
\setlength{\footnotesep}{0in}
\setlength{\topmargin}{-\headsep}
\setlength{\oddsidemargin}{0.25in}
\setlength{\evensidemargin}{0.25in}
\setlength{\parindent}{0.54\baselineskip}
\sloppy\sloppypar
\fi


\def\figurename{Figure}
\def\tablename{Table}
\def\fnum@figure{{\bf\figurename~\thefigure}}
\def\fnum@table{{\bf\tablename~\ifappletter\thesection\fi\thetable}}


\def\tempfootmark#1{}

\newcount\c@affilcount
\renewcommand*\altaffiliation[2][]{%
\@AF@join{\ifanonymous\else
#1#2\ifmodern\baselineskip=14pt\fi
\if@two@col\hsize=.5\textwidth
\advance\hsize by -18pt
\fi\fi}%
}%


\def\correspondingauthor#1{{\ifanonymous
\else
\renewcommand\thefootnote{\hskip-12pt}
\footnote{Corresponding author: #1\ifmodern\vrule depth 5pt
width 0pt\relax\fi}\fi}}

\let\saveemail\email
\def\email#1{\ifanonymous
\else{\let\ltx@footmark\tempfootmark
\saveemail{}}
{\renewcommand\thefootnote{\hskip-12.1pt}
\footnote{\href{mailto: #1}{#1}\ifmodern\vrule depth 7pt width
0pt\relax\else\ifmanu\vskip-4pt\else\vrule depth 7pt width
0pt\fi\fi}}\fi}

\def\nocollaboration#1{%
\collaboration{#1}{\vbox to
0pt{\vss\centerline{---}\vskip2pt}}
}

%% May 19
%% \AuthorsAndCollaboration changed to \FullCollaborationID 
%% June 6 \FullCollaborationID changed to \xcollaboration{}{}
%% june 7 \collaboration changed to \xcollaboration; fullcollaborationid changed to \collaboration

%% here just in case we need it in the future...
\def\xcollaboration#1#2{
\global\advance\allentries by 1
\expandafter\def\csname
currCollabLimit\the\allentries\endcsname{#1}
\@author@def{\@booleantrue
\collaboration@sw}{#2}
}

\let\savelistand\@listand
\newcount\numauthors
\newcount\collabnum
\newbox\collabnamebox
\newif\iffirstcollab
\global\firstcollabtrue

\newif\ifcollaborationon
\def\collaboration#1#2{\global\collaborationontrue
\global\advance\collabnum by 1
\iffirstcollab\global\firstcollabfalse
\expandafter\xdef\csname currCollabLimit0\endcsname{#1}
\fi
%% make this def so that we can use it when we want in doauthor:
\expandafter\gdef\csname specificCollabLimit\the\collabnum\endcsname{#1} 
%% This allows us access the number of authors per collaboration:
\expandafter\xdef\csname
CollabTotalAuthors\the\collabnum\endcsname{\the\entriesinthiscollab} 
\global\entriesinthiscollab=0
%%
\global\advance\allentries by 1
%% 
%% we should have only one of these with this number; this used to say when
%% use specificCollabLimit:
\expandafter\gdef\csname currCollabLimit\the\allentries\endcsname{#1}
%% we should have only one of these with this number
\expandafter\gdef\csname
currCollabName\the\allentries\endcsname{%\ifnum#1>0
%\sc And the\vskip4pt\fi
#2}
%%
\let\doEtAl\relax
%% not here
\@author@def{\@booleanfalse
\collaboration@sw}{}}


\def\and{
\centerline{\vbox {\vrule height 12pt width0pt and\vskip2pt}}
}

%\def\andthe{%
%\collaboration{\vbox {\vrule height 12pt width0pt and
%the\vskip2pt}}\vskip4pt
%}

%% gets rid of () around collaboration
\def\@collaboration@present#1#2#3#4{%
\par
 \begingroup
\vskip3pt
\iflongauthor\vskip-4pt\ifmodern\vskip-6pt\fi\fi
  \frontmatter@collaboration@above
  \@affilID@def{}%
  \@tempcnta\z@
  \@author@present{}{\ignorespaces#3\unskip}{#4}%
%% October 2017
  \par
\ifmodern%\vskip-4pt
\else
\iflongauthor\else\vskip-6pt\fi\fi
 \endgroup
\iflongauthor 
\else\vskip8pt\fi
 \set@listcomma@list#1%
\vskip1pt %% was \vskip9pt
}%


%%% These lines were commented out until a fix could be applied that
%%% addresses the underlying issues. The problem is that on Linux systems
%%% you can not write a "hidden" .bib file. There is no issue with this
%%% on Mac OS X nor Windows.
%%% get rid of \jobname Notes being sent to .aux file:
%\let\bibdata@app\relax
%\def\pre@bibdata{}

%% This is the underlying issue:
%% Redefining \@bibdataout@init will keep useless file, \jobnameNotes, from being written.
\let\@bibdataout@init\relax

%% redefining and simplifying RevTeX4-1 definition, so that bibliography is not looking for
%% \jobnameNotes:

\def\bibliography#1{\bibliography@latex{#1}}%


%%%
\newif\iffrontmatterfirsttime
\global\frontmatterfirsttimetrue
\let\savecentercr\@centercr
\def\frontmatter@author@produce@script{%
  \begingroup
    \let\@author@present\@author@present@script
    \frontmatterverbose@sw{\typeout{\string\frontmatter@author@produce@script:}\say\@AAC@list\say\@AFF@list\say\@AFG@list}{}%
    \let\AU@temp\@empty
    \@tempcnta\z@
    \let\AF@opr \@gobble
    \def\AU@opr{\@author@count\@tempcnta}%
    \def\CO@opr{\@collaboration@count\AU@temp\@tempcnta}%
    \@AAC@list
    \expandafter\CO@opr\@author@cleared
    \begingroup
     \frontmatter@authorformat
     \let\AF@opr \@affilID@def
     \let\AU@opr \@author@present
     \def\CO@opr{\@collaboration@present\AU@temp}%
     \set@listcomma@list\AU@temp
     \@AAC@list
%% \, added to definition taken from revtex4-1 to prevent error message.
\,     \unskip\unskip
     \par
    \endgroup
%% below
    \begingroup
     \frontmatter@above@affiliation@script
     \let\AFF@opr \@affil@script
{\parskip=2pt
\def\@centercr{\vrule depth 3pt width0pt\vskip1sp}
     \@AFF@list}
\let\@centercr\savecentercr
%% sept 14, 2020
\iffrontmatterfirsttime
\frontmatter@footnote@produce
\global\frontmatterfirsttimefalse
\fi
     \par
    \endgroup
  \endgroup
}%

\let\auto@bib\relax

\newif\iffirstaffil
\firstaffiltrue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% These commands were in aastex earlier; I redefined.
\def\tighten{\global\tightentrue\normalsize}
\let\tightenlines=\tighten
\def\singlespace{\par\global\doublespacefalse\global\tightenfalse\normalsize}
\def\doublespace{\par\global\doublespacetrue\global\tightenfalse\normalsize}

\bibliographystyle{aasjournal}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% October 2017

\renewenvironment{widetext}{%
\if@two@col
  \par\ignorespaces
  \setbox\widetext@top\vbox{%
   \hb@xt@\hsize{%
    \leaders\hrule\hfil
    \vrule\@height6\p@
   }%
  }%
  \setbox\widetext@bot\hb@xt@\hsize{%
    \vrule\@depth6\p@
    \leaders\hrule\hfil
  }%
  \onecolumngrid
  \vskip10\p@
  \dimen@\ht\widetext@top\advance\dimen@\dp\widetext@top
  \cleaders\box\widetext@top\vskip\dimen@
  \vskip6\p@
  \prep@math@patch
\fi}{%
\if@two@col
  \par
  \vskip6\p@
  \setbox\widetext@bot\vbox{%
   \hb@xt@\hsize{\hfil\box\widetext@bot}%
  }%
  \dimen@\ht\widetext@bot\advance\dimen@\dp\widetext@bot
  \cleaders\box\widetext@bot\vskip\dimen@
  \vskip8.5\p@
  \twocolumngrid\global\@ignoretrue
  \@endpetrue
\fi}%
\newbox\widetext@top
\newbox\widetext@bot

\urlstyle{rm}

\def\doi#1{\href{https://doi.org/#1}{https://doi.org/#1}}
\let\savedoi\doi

\def\mdash{---}

\newif\ifcenterwidetable
%% these two definitions are the same, but it might
%% be convenient to have both
\def\centerwidetable{\global\centerwidetabletrue}


%% this doesn't seem to be documented, assume we don't need it:
%\def\centernarrowtable{\global\centerwidetablefalse}

%% default, center within width of text on both sides
 \def\LT@LR@c{\LTleft=0pt plus1fill 
  \LTright\LTleft}%

\def\widetable{\def\LT@LR@c{\LTleft=0pt minus1fill 
  \let\LTright\LTleft}}

\newif\ifreturntotwocol


%%%%%%%%%
%% to help with graceful linebreaks in two column text:
  \tolerance 9999%
%% sloppy defines emergencystretch to be 3 em, this is a bit
%% more conservative:
  \emergencystretch 1em 
  \hfuzz .5\p@

%%%%%%%%%
%% Nominal Conversion Constants

%%  \boldmath inside the \hbox ??
\def\nomSolarEffTemp{\leavevmode\hbox{\boldmath$\mathcal{T}^{\rm N}_{\mathrm{eff}\odot}$}}
\def\nomTerrEqRadius{\leavevmode\hbox{\boldmath$\mathcal{R}^{\rm N}_{E\mathrm e}$}}
\def\nomTerrPolarRadius{\leavevmode\hbox{\boldmath$\mathcal{R}^{\rm N}_{E\mathrm p}$}}
\def\nomJovianEqRadius{\leavevmode\hbox{\boldmath$\mathcal{R}^{\rm
N}_{J\mathrm e}$}}
 \def\nomJovianPolarRadius{\leavevmode\hbox{\boldmath$\mathcal{R}^{\rm
 N}_{J\mathrm p}$}}
 \def\nomTerrMass{\leavevmode\hbox{\boldmath$(\mathcal{GM})^{\rm N}_{\mathrm E}$}}
 \def\nomJovianMass{\leavevmode\hbox{\boldmath$(\mathcal{GM})^{\rm N}_{\mathrm J}$}}
 \def\Qnom{\leavevmode\hbox{\boldmath$\mathcal{Q}^{\rm N}_{\odot}$}}
\let\Qn\Qnom

%% Generic commands that can be given an argument:
\def\nom#1{\leavevmode\hbox{\boldmath$\mathcal{#1}^{\rm N}_{\odot}$}}
\def\Eenom#1{\leavevmode\hbox{\boldmath$\mathcal{#1}^{\rm N}_{Ee}$}}
\def\Epnom#1{\leavevmode\hbox{\boldmath$\mathcal{#1}^{\rm N}_{Ep}$}}
\def\Jenom#1{\leavevmode\hbox{\boldmath$\mathcal{#1}^{\rm N}_{Je}$}}
\def\Jpnom#1{\leavevmode\hbox{\boldmath$\mathcal{#1}^{\rm N}_{Jp}$}}

%%%%%%%%%%%%%%%%
%% Ability to have tables, equations, figures in appendix start from 1, and use appendix section letter.

\newif\ifappletter
\def\apptablenumbers{\global\applettertrue
\setcounter{table}{0}
\setcounter{figure}{0}
\setcounter{equation}{0}
\def\thetable{\thesection\the\c@table}%
\def\fnum@table{{\bf\tablename~\thetable}}%
\def\thefigure{\thesection\the\c@figure}%
\def\fnum@figure{{\bf\figurename~\thefigure}}%
}%

%%% easier to remember than \apptablenumbers
\let\restartappendixnumbering\apptablenumbers

\def\resetapptablenumbers{\global\c@table=0
\global\c@figure=0
\global\c@equation=0
\def\thetable{\thesection\the\c@table}
\def\fnum@table{{\bf\tablename~\thetable}}%
\def\thefigure{\thesection\the\c@figure}
\def\fnum@figure{{\bf\figurename~\thefigure}}%
}

%% written for aastex63
\newif\ifallauthors
\def\allauthors{
\global\allauthorstrue
\let\doauthor\doAllauthors
\ifanonymous
\vskip6pt\vskip1sp\centerline{\large\bf All Authors and
Affiliations\vrule depth 18pt width0pt}\nobreak
\centerline{Anonymous author(s)}
\else
\ifnumlines\nolinenumbers\fi
\onecolumngrid
\clearpage
{\vskip6pt\vskip1sp\centerline{\large\bf All Authors and
Affiliations\vrule depth 18pt width0pt}\nobreak
\global\docollabnum=0
\global\totalentries=0
\global\countauthors=0
\maketitle
}\fi}

%% written for AASTeX62
\def\oldallauthors{%% this conditional keeps \allauthors from turning on
%%                 unless \AuthorCollaborationLimit is used:
\ifnumlines\nolinenumbers\fi
\onecolumngrid
\clearpage
\AuthorCollaborationLimit=10000
%\largestAffilNum=10000 <<== not needed
{\vskip6pt\vskip1sp\centerline{\large\bf All Authors and
Affiliations\vrule depth 18pt width0pt}\nobreak
\maketitle
}}
\let\AllAuthors\allauthors


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Highlight Figure
\definecolor{ltblue}{cmyk}{.5,.1,.1,0}
\newdimen\currwidth
\long\def\highlightfigure#1{
\bgroup
\fboxrule=4pt
\fboxsep=12pt
\vskip6pt
\global\currwidth=\hsize
\global\advance\currwidth by -32pt
\noindent\hbox to\currwidth{\fcolorbox{ltblue}{white}{%
$\vcenter{\hbox to\currwidth{\hss#1\hss}}$}}
\egroup}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% New interactive command:

\def\xlc{lc}
\def\xjs{js}
\def\xanimation{animation}
\newbox\interactbox
\newif\ifcorrectentries


\def\interactive#1#2{
\global\correctentriesfalse\def\checkone{#1}
\ifx\checkone\xlc\correctentriestrue\fi
\ifx\checkone\xjs\correctentriestrue\fi
\ifx\checkone\xanimation \correctentriestrue\fi
\def\checktwo{#2}
\ifx\checktwo\empty\global\correctentriesfalse\fi
\ifcorrectentries
\else\hrule height1pt\vskip12pt\bf ERROR:
The command
{\tt\string\begin\string{interactive\string}\string{\string}\string{\string}} 
needs two arguments. The
first describes the interactive type and the only allowed choices are
lc, js, or animation.\\
The second argument is the movie file or package
containing the interactive figure components. See\\
\url{https://journals.aas.org/aastexguide/\hashmark interactivefig} for more
details.
\vskip12pt\hrule height1pt
\fi \global\currwidth\hsize
\global\advance\currwidth by -32pt
%% we need to set \currwidth here because interactbox expands 
%% before highlightfigure
\global\setbox\interactbox=\vbox\bgroup\hsize=\currwidth\relax\centering
}

\def\endinteractive{\egroup
\ifcorrectentries\highlightfigure{\copy\interactbox}\fi
}

%%%% For cases in which footnotes are called in full width environment
%%   but are used in two cols. This keeps them from overwriting the
%%   second column:

\let\savefootnote\footnote
\def\onecolfootnote#1{\savefootnote{\hsize=.5\textwidth\advance\hsize
by-18pt\relax#1}}

\def\onecolumnfootnotes{\let\footnote\onecolfootnote}

\newdimen\movetableright

%% for anonymous option
\newif\ifsuppressAffiliations
\def\suppressAffiliations{\global\suppressAffiliationstrue}

%% new definition to be used in trackchanges:
\def\sout#1{\raise5pt\hbox{\underline{\vbox to
0pt{\vskip-4pt\hbox{#1\vrule height 8.5pt depth 3.5pt width0pt}\vss}}}} 

%% changes for affiliations when anonymous option is used:
\ifanonymous
\suppressAffiliationstrue
\gdef\affiliation#1{}%% may 20, 2020
\def\xauthor[#1]#2{}
\let\yauthor\@gobble
\def\lookforbracket{\ifx\next[
\let\go\xauthor\else\let\go\yauthor\fi\go}
\def\author{\futurelet\next\lookforbracket}
\fi


%%%%%%%%%%%%%
%% Line numbering


\ifnumlines
\usepackage[mathlines]{lineno}%% dec 20
%%
\linenumbers*[1]
%%\linenumbersep default=10pt
\if@two@col
\advance\linenumbersep -6pt
\else
\advance\linenumbersep 12pt
\fi
\fi

%% Use when line numbers are on, and after \usepackage{amsmath}.
%% Will make line numbering work on math as it would without amsmath.
\def\fixmathlinenumbering{
\let\saveopensqbracket\[
\def\[{\linenomath\saveopensqbracket}
\let\saveequation\equation
\let\saveendequation\endequation
%%
 \let\savealign\align
 \let\saveendalign\endalign
%%
 \let\savealigned\aligned
 \let\saveendaligned\endaligned
%%
\def\align{\linenomath\savealign}
 \def\endalign{\saveendalign}
 \def\aligned{\linenomath\savealigned}
 \def\endaligned{\linenomath\saveendaligned}
%%
\def\equation{\linenomath\saveequation}
\def\endequation{\linenomath\saveendequation}
\usepackage[mathlines]{lineno}
}

\AtBeginDocument{\expandafter\ifx\csname
@amsmath@err\endcsname\relax\else\fixmathlinenumbering\fi}

\endinput
