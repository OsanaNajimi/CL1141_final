export default function Architecture() {
    return (
        <section id="architecture" style={{
            minHeight: '80vh',
            width: '100%',
            backgroundColor: '#222', // Light contrast
            color: 'white',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            padding: '8rem 2rem'
        }}>
            <h2 style={{
                fontSize: '3rem',
                fontFamily: 'serif',
                marginBottom: '3rem',
                color: 'white'
            }}>
                System Architecture
            </h2>

            <div style={{
                maxWidth: '800px',
                fontSize: '1.2rem',
                lineHeight: '1.8',
                textAlign: 'center',
                color: 'white'
            }}>
                <p style={{ marginBottom: '2rem' }}>
                    Our system is built on a robust pipeline that integrates phonetic matching, semantic analysis,
                    and cultural filtering.
                </p>
                <p style={{ marginBottom: '2rem' }}>
                    <strong>1. Phonetic Analysis:</strong> We decompose your English name into syllables and finding
                    Chinese characters with similar pronunciations using IPA mapping.
                </p>
                <p style={{ marginBottom: '2rem' }}>
                    <strong>2. Semantic Matching:</strong> We use sentence embeddings to compare your personal description
                    with the meanings of thousands of Chinese characters.
                </p>
                <p>
                    <strong>3. Cultural Scoring:</strong> We rank names based on gender appropriateness, age relevance,
                    and frequency of use to ensure natural-sounding results.
                </p>
            </div>
        </section>
    );
}
