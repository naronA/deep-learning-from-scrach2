package main

import (
	"fmt"
	"strings"
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

func main() {
	text := "You say goodbye and I say helo."
	text = strings.Replace(text, ".", " .", -1)
	fmt.Println(text)
	words := strings.Split(text, " ")
	fmt.Println(words)

	wordToId, idToWord := makeDict(words)

	fmt.Println(wordToId)
	fmt.Println(idToWord)
}
