# Wstęp
Na wstępię wyjaśnię, że kod źródłowy, został mocno zmieniony przez narzędzia AI (ChatGPT 4.5). Oryginalny kod uzupełniłem samemu, ale później, gdy 
zacząłem ekeprymentować z Parametric Study i przepaliłem 24H wielowątkowego przetwarzania na 16 wątkach CPU (Ryzen 7 7800X3D), a nadal moje obliczenia się nie skończyły
(oczywiście nie ciągiem, parę razy zaczynałem trening od nowa, bo gdzieś wyskoczyło dzielenie przez zero, albo dane które zbierałem jednak mnie nie usatysfakcjonowały),
to stwierdziłem, że użyję numby, żeby przyspieszyć obliczenia. Okazało się to strzałem w dziesiątkę bo dla tragicznych przypadków typu `alpha=0.01 i step_size=2` które liczyły się całą noc i nadal się nie doliczyły, to trening zaczął trwać kilka minut. Nawet zakręt D, policzył mi się po poprawkach w około 5 minut. 

Oryginalny kod został przeniesiony do katalogu `problem_origianl`.

# Polityka E-greedy
## Trening zakrętu B

Historia kar treningu zakrętu B z parametrami `alpha=0.3` i `step_size=5`:

<img src="raport_imgs/penalties_b.png" alt="Penalties B" style="max-width: 500px">

Przejazdy ewaluacyjne zakrętu B z tymi parametrami:

<img src="raport_imgs/tracks_b.png" alt="Tracks B" style="max-width: 500px">

## Trening zakrętu C

Historia kar treningu zakrętu C z parametrami `alpha=0.3` i `step_size=5`:

<img src="raport_imgs/penalties_c.png" alt="Penalties C" style="max-width: 500px">

Przejazdy ewaluacyjne zakrętu C z tymi parametrami:

<img src="raport_imgs/tracks_c.png" alt="Tracks C" style="max-width: 500px">

## Studium parametryczne
Jako metrykę oceny parametrów, przyjąłem średnią karę liczoną na podstawie 10 epizodów ewaluacyjnych, liczonych co 10 epizodów treningowych.

Poniżej wykres:

<img src="raport_imgs/parametric_study_plot.png" alt="Parametric Study" style="max-width: 500px">

## Trening zakrętu D

Historia kar treningu zakrętu D z parametrami `alpha=0.68` i `step_size=3` - optymalne znalezione w studium parametrycznym:

<img src="raport_imgs/penalties_d.png" alt="Penalties D" style="max-width: 500px">

Przejazdy ewaluacyjne zakrętu D z tymi parametrami:

<img src="raport_imgs/tracks_d.png" alt="Tracks D" style="max-width: 500px">

# Polityka z przyspieszaniem

Parametry to:
- `alpha=0.68`
- `step_size=3`
- `speeding_rate=0.2`

## Kary podstawowy - zakręt C

<img src="raport_imgs/no_speeding_penalties.png" alt="Kary bez speedingu zakręt C" style="max-width: 500px">

## Kary przyspieszający - zakręt C

<img src="raport_imgs/speeding_penalties.png" alt="Kary ze speedingiem zakręt C" style="max-width: 500px">

## Kary przypieszający bez importance sampling - zakręt C

<img src="raport_imgs/speeding_penalties_no_is.png" alt="Kary ze speedingiem zakręt C bez importance sampling" style="max-width: 500px">

Widać na wykresach, że agenty ze speedingiem mają większy problem z nauczeniem się zakrętu. Widać też, że wyłączenie importance sampling przyspiesza trening
