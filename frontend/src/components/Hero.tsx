import Link from 'next/link';

export default function Hero() {
    return (
        <section style={{
            height: '100vh',
            width: '100%',
            backgroundColor: '#222', // Almost black
            color: 'white',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            textAlign: 'center',
            padding: '0 2rem'
        }}>
            {/* <p style={{
                fontSize: '0.9rem',
                letterSpacing: '0.2em',
                textTransform: 'uppercase',
                marginBottom: '2rem',
                opacity: 0.7
            }}>
                Computational Linguistics Group 7
            </p> */}

            <h1 style={{
                fontSize: 'clamp(3rem, 10vw, 6rem)',
                fontFamily: 'serif',
                fontWeight: 'bold',
                lineHeight: 1.1,
                marginTop: '10vh', // Adjust for sticky nav
                marginBottom: '2rem'
            }}>
                Find Your<br />Chinese Name
            </h1>

            <p style={{
                fontSize: '1.5rem',
                fontWeight: 300,
                opacity: 0.8,
                marginBottom: '4rem',
                maxWidth: '600px'
            }}>
                Not just translation, but reconstruct your identity.
            </p>

            <Link href="/generate" style={{
                padding: '1rem 2.5rem',
                border: '1px solid black',
                borderRadius: '2rem',
                fontSize: '1.2rem',
                letterSpacing: '0.15em',
                textTransform: 'uppercase',
                transition: 'all 0.3s ease',
                backgroundColor: 'gold',
                color: 'black'
            }}>
                Start Naming
            </Link>

            <div style={{
                position: 'absolute',
                bottom: '2rem',
                fontSize: '0.8rem',
                letterSpacing: '0.3em',
                textTransform: 'uppercase',
                opacity: 0.5
            }}>
                Scroll<br />↓
            </div>
        </section>
    );
}
