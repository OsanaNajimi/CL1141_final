'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';

export default function Navbar() {
    const pathname = usePathname();
    const isGeneratePage = pathname === '/generate';

    const [hoveredLink, setHoveredLink] = useState<string | null>(null);

    const linkStyle = (name: string) => ({
        opacity: hoveredLink === name ? 1 : 0.7,
        textDecoration: hoveredLink === name ? 'underline' : 'none',
        textUnderlineOffset: '5px',
        transition: 'opacity 0.2s ease, text-decoration 0.2s ease'
    });

    return (
        <nav style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '1rem 2rem', // Reduced padding slightly
            backgroundColor: 'rgba(17, 17, 17, 1)',
            backdropFilter: 'blur(5px)',
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            zIndex: 100,
            color: 'white',
            borderBottom: '1px solid #333',
            flexWrap: 'wrap', // Allow wrapping
            gap: '1rem' // Gap when wrapped
        }}>
            <Link href="/" style={{ fontSize: '1.5rem', fontWeight: 'bold', letterSpacing: '0.05em', fontFamily: 'serif' }}>
                Chinese Name Generator
            </Link>

            <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.9rem', fontWeight: 500, letterSpacing: '0.15em', flexWrap: 'wrap' }}>
                {isGeneratePage ? (
                    <Link
                        href="/"
                        style={linkStyle('Home')}
                        onMouseEnter={() => setHoveredLink('Home')}
                        onMouseLeave={() => setHoveredLink(null)}
                    >
                        HOME
                    </Link>
                ) : (
                    <>
                        <Link
                            href="/"
                            style={linkStyle('Home')}
                            onMouseEnter={() => setHoveredLink('Home')}
                            onMouseLeave={() => setHoveredLink(null)}
                        >
                            HOME
                        </Link>
                        <Link
                            href="#about"
                            style={linkStyle('About')}
                            onMouseEnter={() => setHoveredLink('About')}
                            onMouseLeave={() => setHoveredLink(null)}
                        >
                            ABOUT
                        </Link>
                        <Link
                            href="#architecture"
                            style={linkStyle('Architecture')}
                            onMouseEnter={() => setHoveredLink('Architecture')}
                            onMouseLeave={() => setHoveredLink(null)}
                        >
                            ARCHITECTURE
                        </Link>
                        <Link
                            href="/generate"
                            style={linkStyle('Get A Name')}
                            onMouseEnter={() => setHoveredLink('Get A Name')}
                            onMouseLeave={() => setHoveredLink(null)}
                        >
                            GET A NAME
                        </Link>
                    </>
                )}
            </div>
        </nav>
    );
}
