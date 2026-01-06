# pure-ccg

Copyright (c) 2019-2026 China University of Water Resources and Electric Power,
All rights reserved.

This is a Chinese syntactic and semantic parser built on pure Combinatory Categorial Grammar.
This implemention was forked from another github repository: my-ccg 0.2.7.0 at 2026-1-6, which is the original version 0.1.0.0. Then, its grammar changed to pure Combinatory Categorial Grammar.

Main functions are as follows:
(1) An interactive syntactic and semantic parser was implemented. Sentential parsing was considered as a process of iterations of syntax type combinations. During every time of iteration, syntax type conversions were selected on demand by user, then syntax type combinations were finished by machine, finally syntactic ambiguities were resolved by user, and the resultant phrase set includes only one parsing tree instead of a syntactic forest.
(2) Introducing Clause Theory, a sentence was considered as a sequence of clauses. When parsing a sentence including multiple clauses, user could select to start parsing from a certain clause, which avoided repeat of analyzing the formerly parsed clauses. Using scripts to store selections of non-classical syntactic types and removed phrases, every clause might be parsed again automatically.
(3) A marking system of categorial conversions had already been formed, which included various non-classical syntactic types used by parts of speech and phrasal structures. By augmenting with categorial conversions, Combinatory Categorial Grammar could express inflectional-absent Chinese.
(4) Combinatory Logic was implemented, and CL terms served as semantic representations instead of lambda-terms.
(5) Disambiguation on categorial conversions and syntactic structures were implemented as machine algorithms, such that the whole sentential structural parsing could be done without human interventions.
(6) The repeatable parsing of sentential structural was implemented based upon scripts, which were created during human-machine interactive parsing.
(7) Most important, this category grammar completely comes from CL, in which categorial symbols do not any more include backward slashes, all rules are defined by CL terms.

The current version is 0.1.0.0.
