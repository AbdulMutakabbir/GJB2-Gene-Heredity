import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # init joint_probability
    probability = 1

    # loop over people to determine joint probability
    for person in people:
        # get person's gene
        person_gene = 2 if person in two_genes else 1 if person in one_gene else 0

        # check if person has trait
        person_trait = True if person in have_trait else False

        # calculate the probability that the person has traits
        probability *= PROBS["trait"][person_gene][person_trait]

        # init parents
        parents = {"mother": people[person]["mother"], "father": people[person]["father"]}

        # Check for parents
        if parents["mother"] is None and parents["father"] is None:
            # calculate probability form know data
            probability *= PROBS["gene"][person_gene]
        # else calculate based on parents
        else:
            # init pass_probability
            pass_probability = []

            # loop over parents
            for parent in parents:
                # calculate probability of parent passing gene
                # if parent has 2 gene then 100% probability - probability of mutation
                # if parent has 1 gene then  50% probability
                # if parent has 0 gene then   0% probability + probability of mutation
                parent_pass_probability = (
                    1 - PROBS["mutation"] if parents[parent] in two_genes else
                    0.5 if parents[parent] in one_gene else
                    PROBS["mutation"]
                )

                # append parent probability
                pass_probability.append(parent_pass_probability)

            # calculate probability for receiving form parents
            # if person has 2 gene then probability of both parents passing gene
            # if person has 1 gene then probability of mother passing gene and not father or probability of father passing the gene and not mother
            # if person has 0 gene then probability of not getting the gene form both parents
            probability *= (
                pass_probability[0] * pass_probability[1] if person_gene == 2 else
                (pass_probability[0] * (1 - pass_probability[1])) + (pass_probability[1] * (1 - pass_probability[0])) if person_gene == 1 else
                ((1 - pass_probability[0]) * (1 - pass_probability[1]))
            )

    # return joint probability
    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # loop over people in probabilities dictionary
    for person in probabilities:
        # get person's gene
        person_gene = 2 if person in two_genes else 1 if person in one_gene else 0

        # check if person has trait
        person_trait = True if person in have_trait else False

        # added person gene probability
        probabilities[person]["gene"][person_gene] += p

        # add person trait probability
        probabilities[person]["trait"][person_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    # loop over people in probabilities dictionary
    for person in probabilities:
        # get person's gene and trait probabilities
        person_gene_probabilities, person_trait_probabilities = probabilities[person].values()

        # sum genes probabilities
        person_gene_probabilities_sum = sum(person_gene_probabilities.values())

        # sum traits probabilities
        person_trait_probabilities_sum = sum(person_trait_probabilities.values())

        # update genes
        for gene in probabilities[person]["gene"]:
            probabilities[person]["gene"][gene] /= person_gene_probabilities_sum

        # update traits
        for trait in probabilities[person]["trait"]:
            probabilities[person]["trait"][trait] /= person_trait_probabilities_sum


if __name__ == "__main__":
    main()
