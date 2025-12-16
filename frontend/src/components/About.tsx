export default function About() {
    return (
        <section id="about" style={{
            minHeight: '80vh',
            width: '100%',
            backgroundColor: '#222',
            color: 'white',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            padding: '8rem 2rem',
        }}>
            <h2 style={{
                fontSize: '3rem',
                fontFamily: 'serif',
                marginBottom: '3rem',
                color: 'white'
            }}>
                About Our Project
            </h2>

            <div style={{
                maxWidth: '800px',
                fontSize: '1.2rem',
                lineHeight: '1.8',
                opacity: 0.8,
                textAlign: 'center'
            }}>
                <p style={{ marginBottom: '2rem' }}>
                    <strong>"Chinese Name Generator"</strong> is an intelligent system designed to reconstruct your identity in a new language.
                    We believe a name should carry the weight of your personality, the melody of your original name, and the cultural depth of the Chinese language.
                </p>
                <p style={{ marginBottom: '3rem' }}>
                    Our algorithm considers four key dimensions to craft your unique name:
                </p>

                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                    gap: '2rem',
                    marginBottom: '3rem',
                    textAlign: 'left'
                }}>
                    {/* Card 1: Meaning */}
                    <div style={{
                        padding: '1.5rem',
                        backgroundColor: '#333',
                        borderRadius: '8px',
                        border: '1px solid #444',
                        boxShadow: '0 4px 6px rgba(0,0,0,0.3)'
                    }}>
                        <h3 style={{ marginBottom: '0.5rem', color: '#fff', fontSize: '1.3rem' }}>Meaning</h3>
                        <p style={{ fontSize: '1rem', color: '#ccc' }}>
                            AI-powered semantic analysis matches your personality description with the deep meanings of thousands of Chinese characters.
                        </p>
                    </div>

                    {/* Card 2: Sound */}
                    <div style={{
                        padding: '1.5rem',
                        backgroundColor: '#333',
                        borderRadius: '8px',
                        border: '1px solid #444',
                        boxShadow: '0 4px 6px rgba(0,0,0,0.3)'
                    }}>
                        <h3 style={{ marginBottom: '0.5rem', color: '#fff', fontSize: '1.3rem' }}>Sound</h3>
                        <p style={{ fontSize: '1rem', color: '#ccc' }}>
                            Advanced phonetic algorithms analyze your English name to find Chinese characters that echo its original melody.
                        </p>
                    </div>

                    {/* Card 3: Demographics */}
                    <div style={{
                        padding: '1.5rem',
                        backgroundColor: '#333',
                        borderRadius: '8px',
                        border: '1px solid #444',
                        boxShadow: '0 4px 6px rgba(0,0,0,0.3)'
                    }}>
                        <h3 style={{ marginBottom: '0.5rem', color: '#fff', fontSize: '1.3rem' }}>Demographics</h3>
                        <p style={{ fontSize: '1rem', color: '#ccc' }}>
                            Gender and age-based scoring techniques ensure your generated name feels natural, appropriate, and authentic.
                        </p>
                    </div>

                    {/* Card 4: Tone */}
                    <div style={{
                        padding: '1.5rem',
                        backgroundColor: '#333',
                        borderRadius: '8px',
                        border: '1px solid #444',
                        boxShadow: '0 4px 6px rgba(0,0,0,0.3)'
                    }}>
                        <h3 style={{ marginBottom: '0.5rem', color: '#fff', fontSize: '1.3rem' }}>Tone Flow</h3>
                        <p style={{ fontSize: '1rem', color: '#ccc' }}>
                            We analyze tone combinations to filter out awkward phrasing and ensure your name sounds rhythmic and harmonious.
                        </p>
                    </div>
                </div>

                <p>
                    With advanced controls, you can fine-tune the balance between sound, meaning, and creativity to find a name that truly belongs to you.
                </p>
            </div>
        </section>
    );
}
