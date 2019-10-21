package main

import (
	"strings"

	"gonum.org/v1/gonum/mat"
)

func makeDict(words []string) (map[string]int, map[int]string) {
	wordToId := make(map[string]int)
	idToWord := make(map[int]string)

	for _, w := range words {
		if _, ok := wordToId[w]; !ok {
			newId := len(wordToId)
			wordToId[w] = newId
			idToWord[newId] = w
		}
	}
	return wordToId, idToWord
}

func Preprocess(text string) (*mat.Dense, map[string]int, map[int]string) {
	text = strings.ToLower(text)
	text = strings.ReplaceAll(text, ".", " .")
	words := strings.Split(text, " ")
	wordToID := map[string]int{}
	idToWord := map[int]string{}
	wordIds := []float64{}
	for _, word := range words {
		if _, ok := wordToID[word]; !ok {
			newId := len(wordToID)
			wordToID[word] = newId
			idToWord[newId] = word
		}
		wordIds = append(wordIds, float64(wordToID[word]))
	}
	corpus := mat.NewDense(1, len(wordIds), wordIds)
	return corpus, wordToID, idToWord

}

func CreateCoMatrix(corpus *mat.Dense, vocabSize int, windowSize int) *mat.Dense {
	row := corpus.RawMatrix().Data
	corpusSize := len(row)
	coMatrix := mat.NewDense(vocabSize, vocabSize, nil)
	for idx, wordID := range row {
		for i := 0; i < windowSize+1; i++ {
			leftIdx := idx - i
			rightIdx := idx + i
			if leftIdx >= 0 {
				leftWordID := int(row[leftIdx])
				val := coMatrix.At(int(wordID), leftWordID)
				coMatrix.Set(int(wordID), leftWordID, val+1)
			}
			if rightIdx < corpusSize {
				rightWordID := int(row[rightIdx])
				val := coMatrix.At(int(wordID), rightWordID)
				coMatrix.Set(int(wordID), rightWordID, val+1)
			}
		}
	}
	return coMatrix
}

func CosSimilarity(x *mat.Dense, y *mat.Dense, eps float64) float64 {
	// pow := mat.NewDense(x.RawMatrix().Rows, x.RawMatrix().Cols, nil)
	// pow.Pow(x, 2)
	// sum := mat.Sum(pow)

	// ep := mat.NewDense(x.RawMatrix().Rows, x.RawMatrix().Cols, eps)
	// ep.Add(sum)
}
