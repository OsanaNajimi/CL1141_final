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
                    This project aims to bridge the cultural gap by providing meaningful Chinese names to foreigners.
                    It is not just about translation, but about understanding the essence of your identity and finding
                    characters that resonate with who you are.
                </p>
                <p>
                    Using advanced computational linguistics techniques, we analyze the phonetics, meaning, and cultural
                    nuances of your English name and personal description to generate names that are both phonetically
                    accurate and semantically profound.
                </p>
            </div>
        </section>
    );
}
