# GJB2-Gene-Heredity
Using Bayesian Network and Markov Assumption to determine the probaility of an individual showing traits of deafness and determining the probability of that human having 0, 1 or 2 copies for the GJB2 Gene in them.

# Introduction
Mutated versions of the GJB2 gene are one of the leading causes of hearing impairment in newborns. Each person carries two versions of the gene, so each person has the potential to possess either 0, 1, or 2 copies of the hearing impairment version GJB2. Unless a person undergoes genetic testing, though, it’s not so easy to know how many copies of mutated GJB2 a person has. This is some “hidden state”: information that has an effect that we can observe (hearing impairment), but that we don’t necessarily directly know. After all, some people might have 1 or 2 copies of mutated GJB2 but not exhibit hearing impairment, while others might have no copies of mutated GJB2 yet still exhibit hearing impairment.

Every child inherits one copy of the GJB2 gene from each of their parents. If a parent has two copies of the mutated gene, then they will pass the mutated gene on to the child; if a parent has no copies of the mutated gene, then they will not pass the mutated gene on to the child; and if a parent has one copy of the mutated gene, then the gene is passed on to the child with probability 0.5. After a gene is passed on, though, it has some probability of undergoing additional mutation: changing from a version of the gene that causes hearing impairment to a version that doesn’t, or vice versa.

We can attempt to model all of these relationships by forming a Bayesian Network of all the relevant variables, as in the one below, which considers a family of two parents and a single child.

![Bayesian Network for GJB2](https://github.com/AbdulMutakabbir/GJB2-Gene-Heredity/blob/main/assets/gene_network.png)

Each person in the family has a Gene random variable representing how many copies of a particular gene a person has: a value that is 0, 1, or 2. Each person in the family also has a Trait random variable, which is yes or no depending on whether that person expresses a trait ie hearing imparement based on that gene. There’s an arrow from each person’s Gene variable to their Trait variable to encode the idea that a person’s genes affect the probability that they have a particular trait. Meanwhile, there’s also an arrow from both the mother and father’s Gene random variable to their child’s Gene random variable: the child’s genes are dependent on the genes of their parents.


# Usage
```
> Clone the Git Reposoratory
> cd into "GJB2-Gene-Heredity" directory
> python heredity.py <path-to-dataset>
```

# Output
``` python
$ python heredity.py data/family.csv
Harry:
  Gene:
    2: 0.0092
    1: 0.4557
    0: 0.5351
  Trait:
    True: 0.2665
    False: 0.7335
James:
  Gene:
    2: 0.1976
    1: 0.5106
    0: 0.2918
  Trait:
    True: 1.0000
    False: 0.0000
Lily:
  Gene:
    2: 0.0036
    1: 0.0136
    0: 0.9827
  Trait:
    True: 0.0000
    False: 1.0000
```

``` python
$ python heredity.py data/family2.csv
Arthur:
  Gene:
    2: 0.0147
    1: 0.0344
    0: 0.9509
  Trait:
    True: 0.0000
    False: 1.0000
Hermione:
  Gene:
    2: 0.0608
    1: 0.1203
    0: 0.8189
  Trait:
    True: 0.0000
    False: 1.0000
Molly:
  Gene:
    2: 0.0404
    1: 0.0744
    0: 0.8852
  Trait:
    True: 0.0768
    False: 0.9232
Ron:
  Gene:
    2: 0.0043
    1: 0.2149
    0: 0.7808
  Trait:
    True: 0.0000
    False: 1.0000
Rose:
  Gene:
    2: 0.0088
    1: 0.7022
    0: 0.2890
  Trait:
    True: 1.0000
    False: 0.0000

```

# Technologies
* Python
