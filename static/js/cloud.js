function highlightPhrase(phrase, text, phrases) {
    const url = new URL('/highlight_phrase', window.location.origin);
    url.searchParams.append('text', text);
    url.searchParams.append('phrase', phrase);
    phrases.forEach((p) => url.searchParams.append('phrases', p));
    window.location.href = url.toString();
}
