from lucy.rag.chunking import chunk_text


def test_empty_text_produces_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_produces_single_chunk():
    text = "the quick brown fox"
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert chunks == [text]


def test_long_text_produces_overlapping_chunks():
    words = [f"word{i}" for i in range(500)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=200, overlap=40)

    assert len(chunks) > 1
    # every word appears somewhere in the chunked output
    all_chunked_words = " ".join(chunks).split()
    assert set(words).issubset(set(all_chunked_words))


def test_consecutive_chunks_overlap():
    words = [f"word{i}" for i in range(250)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=200, overlap=40)

    first_words = chunks[0].split()
    second_words = chunks[1].split()
    # the last `overlap` words of chunk 1 should be the first words of chunk 2
    assert first_words[-40:] == second_words[:40]


def test_no_chunk_exceeds_chunk_size_words():
    words = [f"word{i}" for i in range(1000)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=200, overlap=40)

    for chunk in chunks:
        assert len(chunk.split()) <= 200
