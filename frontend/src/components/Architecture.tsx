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
                maxWidth: '1000px',
                fontSize: '1.1rem',
                lineHeight: '1.8',
                color: 'white'
            }}>
                <p style={{ marginBottom: '3rem', textAlign: 'center' }}>
                    Our system uses a sophisticated pipeline to transform your inputs into meaningful Chinese names.
                </p>

                {/* Scoring Section */}
                <h3 style={{ borderBottom: '1px solid #444', paddingBottom: '0.5rem', marginBottom: '1.5rem', fontSize: '1.5rem' }}>
                    1. Calculating the Perfect Score
                </h3>
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                    gap: '1.5rem',
                    marginBottom: '3rem'
                }}>
                    <div style={{ padding: '1.5rem', backgroundColor: '#333', borderRadius: '8px', border: '1px solid #444' }}>
                        <h4 style={{ color: '#fff', marginBottom: '0.5rem' }}>🤖 Semantic Embedding</h4>
                        <p style={{ fontSize: '0.95rem', color: '#ccc' }}>
                            We encode your description using <code>paraphrase-multilingual-MiniLM-L12-v2</code> and calculate the cosine similarity against character meanings to find the best conceptual match.
                        </p>
                    </div>
                    <div style={{ padding: '1.5rem', backgroundColor: '#333', borderRadius: '8px', border: '1px solid #444' }}>
                        <h4 style={{ color: '#fff', marginBottom: '0.5rem' }}>🔤 Phonetic Distance</h4>
                        <p style={{ fontSize: '0.95rem', color: '#ccc' }}>
                            Your English name is converted to IPA and syllabified. We calculate the <strong>Manhattan distance</strong> on a 24-dimensional feature space (via <code>panphon</code>) to compare with Chinese characters.
                        </p>
                    </div>
                    <div style={{ padding: '1.5rem', backgroundColor: '#333', borderRadius: '8px', border: '1px solid #444' }}>
                        <h4 style={{ color: '#fff', marginBottom: '0.5rem' }}>📉 Real-World Demographics</h4>
                        <p style={{ fontSize: '0.95rem', color: '#ccc' }}>
                            We calculate gender and age scores based on statistical frequency analysis of real-world name data, ensuring your name fits your desired profile.
                        </p>
                    </div>
                    <div style={{ padding: '1.5rem', backgroundColor: '#333', borderRadius: '8px', border: '1px solid #444' }}>
                        <h4 style={{ color: '#fff', marginBottom: '0.5rem' }}>🎚️ Dynamic Weighting</h4>
                        <p style={{ fontSize: '0.95rem', color: '#ccc' }}>
                            All individual scores are combined using the custom weights you set in the Advanced Options, giving you full control over the prioritization.
                        </p>
                    </div>
                </div>

                {/* Generation Section */}
                <h3 style={{ borderBottom: '1px solid #444', paddingBottom: '0.5rem', marginBottom: '1.5rem', fontSize: '1.5rem' }}>
                    2. Evaluating Final Suggestions
                </h3>
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
                    gap: '1.5rem',
                    marginBottom: '2rem'
                }}>
                    <div style={{ padding: '1.5rem', backgroundColor: '#333', borderRadius: '8px', border: '1px solid #444' }}>
                        <h4 style={{ color: '#fff', marginBottom: '0.5rem' }}>Single Character Selection</h4>
                        <p style={{ fontSize: '0.95rem', color: '#ccc' }}>
                            We rank the entire database by the Total Weighted Score. Experiences show that the top 30 filtered characters usually contain the best candidates for your name.
                        </p>
                    </div>
                    <div style={{ padding: '1.5rem', backgroundColor: '#333', borderRadius: '8px', border: '1px solid #444' }}>
                        <h4 style={{ color: '#fff', marginBottom: '0.5rem' }}>Full Name Generation</h4>
                        <p style={{ fontSize: '0.95rem', color: '#ccc' }}>
                            We combine your chosen Family Name (or a generated one) with high-scoring characters. We then apply a <strong>Tone Filter</strong> to ensure the 3-character combination adheres to euphonic patterns commonly found in native names.
                        </p>
                    </div>
                </div>
            </div>
        </section>
    );
}
