'use client';
import { useState } from 'react';
import Navbar from '@/components/Navbar';
import InputForm from '@/components/InputForm';
import Results from '@/components/Results';

export default function GeneratePage() {
    const [results, setResults] = useState([]);
    const [topCharacters, setTopCharacters] = useState([]);
    const [isLoading, setIsLoading] = useState(false);

    const handleGenerate = async (formData: any) => {
        setIsLoading(true);
        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData),
            });
            const data = await response.json();

            if (data.recommendations) {
                setResults(data.recommendations);
            }
            if (data.top_characters) {
                setTopCharacters(data.top_characters);
            }

        } catch (error) {
            console.error("Error generating names:", error);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <main style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: '#111' }}>
            <Navbar />

            {/* Content Container (Grid or Flex) - adjusted for fixed navbar */}
            <div style={{ display: 'flex', flex: 1, overflow: 'hidden', paddingTop: '85px' }}>
                {/* Left Side: Input Form */}
                <div style={{
                    width: '400px',
                    padding: '2rem',
                    borderRight: '1px solid #333',
                    backgroundColor: '#111',
                    display: 'flex',
                    flexDirection: 'column',
                    // justifyContent: 'center' // Align to top is better for long forms
                }}>
                    <h2 style={{ marginBottom: '2rem', fontSize: '1.5rem', color: 'white', fontFamily: 'serif' }}>Your Details</h2>
                    <InputForm onSubmit={handleGenerate} isLoading={isLoading} />
                </div>

                {/* Right Side: Results */}
                <div style={{
                    flex: 1,
                    padding: '3rem',
                    overflowY: 'auto',
                    backgroundColor: '#111'
                }}>
                    {results.length > 0 ? (
                        <>
                            {/* <h2 style={{ marginBottom: '2rem', fontSize: '1.5rem', color: 'white', fontFamily: 'serif' }}>Generated Names</h2> */}
                            {/* Header moved inside Results as tabs */}
                            <Results results={results} topCharacters={topCharacters} />
                        </>
                    ) : (
                        <div style={{
                            height: '100%',
                            display: 'flex',
                            justifyContent: 'center',
                            alignItems: 'center',
                            color: '#444',
                            fontSize: '1.2rem',
                            flexDirection: 'column',
                            gap: '1rem'
                        }}>
                            <div style={{ fontSize: '3rem', opacity: 0.2 }}>M</div>
                            <div>Fill out the form to start</div>
                        </div>
                    )}
                </div>
            </div>
        </main>
    );
}
