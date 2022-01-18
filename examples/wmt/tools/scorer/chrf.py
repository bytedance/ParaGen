import argparse
import string
from collections import defaultdict


def separate_characters(line):
    return list(line.strip().replace(" ", ""))


def separate_punctuation(line):
    words = line.strip().split()
    tokenized = []
    for w in words:
        if len(w) == 1:
            tokenized.append(w)
        else:
            lastChar = w[-1]
            firstChar = w[0]
            if lastChar in string.punctuation:
                tokenized += [w[:-1], lastChar]
            elif firstChar in string.punctuation:
                tokenized += [firstChar, w[1:]]
            else:
                tokenized.append(w)

    return tokenized


def ngram_counts(wordList, order):
    counts = defaultdict(lambda: defaultdict(float))
    nWords = len(wordList)
    for i in range(nWords):
        for j in range(1, order + 1):
            if i + j <= nWords:
                ngram = tuple(wordList[i:i + j])
                counts[j - 1][ngram] += 1

    return counts


def ngram_matches(ref_ngrams, hyp_ngrams):
    matchingNgramCount = defaultdict(float)
    totalRefNgramCount = defaultdict(float)
    totalHypNgramCount = defaultdict(float)

    for order in ref_ngrams:
        for ngram in hyp_ngrams[order]:
            totalHypNgramCount[order] += hyp_ngrams[order][ngram]
        for ngram in ref_ngrams[order]:
            totalRefNgramCount[order] += ref_ngrams[order][ngram]
            if ngram in hyp_ngrams[order]:
                matchingNgramCount[order] += min(ref_ngrams[order][ngram], hyp_ngrams[order][ngram])

    return matchingNgramCount, totalRefNgramCount, totalHypNgramCount


def ngram_precrecf(matching, reflen, hyplen, beta):
    ngramPrec = defaultdict(float)
    ngramRec = defaultdict(float)
    ngramF = defaultdict(float)

    factor = beta**2

    for order in matching:
        if hyplen[order] > 0:
            ngramPrec[order] = matching[order] / hyplen[order]
        else:
            ngramPrec[order] = 1e-16
        if reflen[order] > 0:
            ngramRec[order] = matching[order] / reflen[order]
        else:
            ngramRec[order] = 1e-16
        denom = factor * ngramPrec[order] + ngramRec[order]
        if denom > 0:
            ngramF[order] = (1 + factor) * ngramPrec[order] * ngramRec[order] / denom
        else:
            ngramF[order] = 1e-16

    return ngramF, ngramRec, ngramPrec


def computeChrF(fpRef, fpHyp, nworder, ncorder, beta, sentence_level_scores=None):
    norder = float(nworder + ncorder)

    # initialisation of document level scores
    totalMatchingCount = defaultdict(float)
    totalRefCount = defaultdict(float)
    totalHypCount = defaultdict(float)
    totalChrMatchingCount = defaultdict(float)
    totalChrRefCount = defaultdict(float)
    totalChrHypCount = defaultdict(float)
    averageTotalF = 0.0

    nsent = 0
    for hline, rline in zip(fpHyp, fpRef):
        nsent += 1

        # preparation for multiple references
        maxF = -100
        bestWordMatchingCount = None
        bestCharMatchingCount = None

        hypNgramCounts = ngram_counts(separate_punctuation(hline), nworder)
        hypChrNgramCounts = ngram_counts(separate_characters(hline), ncorder)

        # going through multiple references

        refs = rline.split("*#")

        for ref in refs:
            refNgramCounts = ngram_counts(separate_punctuation(ref), nworder)
            refChrNgramCounts = ngram_counts(separate_characters(ref), ncorder)

            # number of overlapping n-grams, total number of ref n-grams, total number of hyp n-grams
            matchingNgramCounts, totalRefNgramCount, totalHypNgramCount = ngram_matches(refNgramCounts, hypNgramCounts)
            matchingChrNgramCounts, totalChrRefNgramCount, totalChrHypNgramCount = ngram_matches(
                refChrNgramCounts, hypChrNgramCounts)

            # n-gram f-scores, recalls and precisions
            ngramF, ngramRec, ngramPrec = ngram_precrecf(
                matchingNgramCounts, totalRefNgramCount, totalHypNgramCount, beta)
            chrNgramF, chrNgramRec, chrNgramPrec = ngram_precrecf(
                matchingChrNgramCounts, totalChrRefNgramCount, totalChrHypNgramCount, beta)

            sentRec = (sum(chrNgramRec.values()) + sum(ngramRec.values())) / norder
            sentPrec = (sum(chrNgramPrec.values()) + sum(ngramPrec.values())) / norder
            sentF = (sum(chrNgramF.values()) + sum(ngramF.values())) / norder

            if sentF > maxF:
                maxF = sentF
                bestMatchingCount = matchingNgramCounts
                bestRefCount = totalRefNgramCount
                bestHypCount = totalHypNgramCount
                bestChrMatchingCount = matchingChrNgramCounts
                bestChrRefCount = totalChrRefNgramCount
                bestChrHypCount = totalChrHypNgramCount
        # all the references are done

        # write sentence level scores
        if sentence_level_scores:
            sentence_level_scores.write("%i::c%i+w%i-F%i\t%.4f\n" % (nsent, ncorder, nworder, beta, 100 * maxF))

        # collect document level ngram counts
        for order in range(nworder):
            totalMatchingCount[order] += bestMatchingCount[order]
            totalRefCount[order] += bestRefCount[order]
            totalHypCount[order] += bestHypCount[order]
        for order in range(ncorder):
            totalChrMatchingCount[order] += bestChrMatchingCount[order]
            totalChrRefCount[order] += bestChrRefCount[order]
            totalChrHypCount[order] += bestChrHypCount[order]

        averageTotalF += maxF

    # all sentences are done

    # total precision, recall and F (aritmetic mean of all ngrams)
    totalNgramF, totalNgramRec, totalNgramPrec = ngram_precrecf(totalMatchingCount, totalRefCount, totalHypCount, beta)
    totalChrNgramF, totalChrNgramRec, totalChrNgramPrec = ngram_precrecf(
        totalChrMatchingCount, totalChrRefCount, totalChrHypCount, beta)

    totalF = (sum(totalChrNgramF.values()) + sum(totalNgramF.values())) / norder
    averageTotalF = averageTotalF / nsent
    totalRec = (sum(totalChrNgramRec.values()) + sum(totalNgramRec.values())) / norder
    totalPrec = (sum(totalChrNgramPrec.values()) + sum(totalNgramPrec.values())) / norder

    return totalF, averageTotalF, totalPrec, totalRec


def scoreChrF(hypo, ref):
    tot, cnt = 0, 0
    for r in ref:
        tot += [0]
        cnt += 1
    return tot / cnt


parser = argparse.ArgumentParser()
parser.add_argument('--hypo_filename', metavar='N', type=str, help='hypo_filename')
parser.add_argument('--out_filename', metavar='N', type=str, help='out_filename')
parser.add_argument('--num_candidates', type=int, help="num_candidates")
args, unknown = parser.parse_known_args()

with open(args.hypo_filename, 'r') as fhypo, open(args.out_filename, 'w') as fout:
    max_bleu = 0
    buffer = []
    for i, hypo in enumerate(fhypo):
        buffer.append(hypo)
        if i % args.num_candidates == args.num_candidates - 1:
            for i, h in enumerate(buffer):
                score = computeChrF(["*#".join([r for r in buffer if r != h])], [h], nworder=2, ncorder=6, beta=2)[0]
                fout.write(f'{score / 100}\n')
            buffer.clear()
