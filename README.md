# Data-Driven Enzyme Engineering Hackathon

Enzymes are natural molecular machines that can perform the amazing feat of enabling and accelerating chemical reactions without themselves being consumed in the process. All living organisms on Earth depend on enzymes to process nutrients, to grow and to reproduce. Thanks to the development of modern biotechnology, enzymes can also be produced in large quantities and employed to speed up all kinds of industrially important reactions, ranging from stain removal by detergents to production of food, feed, and biomaterials.

As our requirements for lower resource use get more demanding, we continuously need to improve our enzymes to make them more stable and effective. At DuPont Industrial Biosciences, we routinely screen thousands of enzymes to obtain models of the relationship between the sequence of amino acid residues that make up the enzyme, its 3D structure, and the observed performance.

In this hackathon, we will provide representative data for our challenge, describe the problem and discuss different approaches of how to employ techniques like deep learning to support enzyme optimization process. Join and help us design the enzymes of tomorrow! 

## Installation

None of the packages used in this repository are required for participating in the hackathon but if you want to follow the examples you can do by first installing pipenv

```
pip install pipenv
```

And then do

```
pipenv install
```

After that you should be able to open the notebook

```
jupyter notebook hackathon.ipynb
```

## Submissions

Submitting sequences for the virtual lab facility is done by

```
import requests

sequences = [('AQSVPWGISRVQAPAAHNRGLTGSGVKVAVLDTGISTHPDLNIRRGGASFVPGEPSTQDGNGHGTHVA'
             'GTIAALNNSIGVLGVAPSAELLYAVKVLGASGSSGGSSVSSIAQGLEWAGNNGMHVANLSLGSPSPSA'
             'TLEQAVNSATSRGVLVVAASGNSGAGSISYPARYANAMAVGATDQNNNRASFSQYGAGLDIVAPGVNV'
             'QSTYPGSTAASLNGTSMATPHVAGAAALVKQKNPSWSNVQIRNHLKNTATSLGGSSTTNNLYGSGLVA'
             'AEAATR'),
             ('AQSVPWGASRVQAPAAHNRGLTGSGVAVAVLDTGISTHPDLNIRRGGASFVPGEPSTQDGNGHGTHVA'
             'GTIAALNNSIGVLGVAPSAELLYAVKVLGASGSSGGSSVSSIAQGLEWAGNNGMHVANLSLGSPSPSA'
             'TLEQAVNSATSRGVLVVAASGNSGAGSISYPARYANAMAVGATDQNNNRASFSQYGAGLDIVAPGVNV'
             'QSTYPGSTYASLNGTSMATPHVAGAAALVKQKNPSWSNVQIRNHLKNTATSLGGSSTTNNLYGSGLVN'
             'AEAATR')]

payload = {'team': 'footeam', 'challenge': 0, 'sequences': sequences}

r = requests.post('http://localhost/score', json=payload)
```
