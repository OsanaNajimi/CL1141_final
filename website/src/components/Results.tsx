'use client';
import { useState } from 'react';

type NameResult = {
    name: string;
    family_name: string;
    given_name: string;
    total_score: number;
    details: {
        char1: CharDetails;
        char2: CharDetails;
    }
};

type CharDetails = {
    char: string;
    meaning: number;
    gender: number;
    age: number;
    freq: number;
    phone: number;
};

type TopCharacter = {
    character: string;
    total_score: number;
    details: {
        meaning: number;
        gender: number;
        age: number;
        freq: number;
        phone: number;
    };
};

export default function Results({ results, topCharacters }: { results: NameResult[], topCharacters: TopCharacter[] }) {
    const [activeTab, setActiveTab] = useState<'names' | 'characters'>('names');

    // Only render if we have data for the active view
    if (!results || results.length === 0) return null;

    const TabButton = ({ id, label }: { id: 'names' | 'characters', label: string }) => (
        <button
            onClick={() => setActiveTab(id)}
            style={{
                background: 'transparent',
                border: 'none',
                borderBottom: activeTab === id ? '2px solid white' : '2px solid transparent',
                color: activeTab === id ? 'white' : '#888',
                padding: '0.5rem 1rem',
                fontSize: '1rem',
                cursor: 'pointer',
                transition: 'all 0.2s',
                fontWeight: activeTab === id ? 'bold' : 'normal',
                marginRight: '1rem'
            }}
        >
            {label}
        </button>
    );

    return (
        <div style={{ width: '100%' }}>
            {/* Tabs Header */}
            <div style={{ marginBottom: '2rem', borderBottom: '1px solid #333' }}>
                <TabButton id="names" label="Suggested Full Names" />
                <TabButton id="characters" label="Suggested Characters" />
            </div>

            {/* Content Area */}
            {activeTab === 'names' && (
                <div style={{ width: '100%', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1.5rem' }}>
                    {results.map((item, idx) => (
                        <div key={idx} style={{
                            border: '1px solid #333',
                            borderRadius: '8px',
                            backgroundColor: '#1a1a1a', // Dark card bg
                            padding: '1.5rem',
                            display: 'flex',
                            flexDirection: 'column',
                            boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
                            color: '#ffffff'
                        }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '1rem' }}>
                                <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>
                                    {item.name}
                                </div>
                                <div style={{ fontSize: '0.85rem', color: '#888', fontWeight: 600 }}>
                                    Score: {item.total_score.toFixed(1)}
                                </div>
                            </div>

                            <div style={{ flex: 1 }}>
                                {/* Breakdown */}
                                <div style={{ fontSize: '0.8rem', color: '#aaa', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <span>Meaning:</span>
                                        <span>{(item.details.char1.meaning + item.details.char2.meaning).toFixed(2)}</span>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <span>Phonetic:</span>
                                        <span>{(item.details.char1.phone + item.details.char2.phone).toFixed(2)}</span>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <span>Gender/Age:</span>
                                        <span>{(item.details.char1.gender + item.details.char2.gender + item.details.char1.age + item.details.char2.age).toFixed(2)}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {activeTab === 'characters' && (
                <div style={{ width: '100%', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '1rem' }}>
                    {topCharacters && topCharacters.map((char, idx) => (
                        <div key={idx} style={{
                            border: '1px solid #333',
                            borderRadius: '8px',
                            backgroundColor: '#1a1a1a',
                            padding: '1rem',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            textAlign: 'center',
                            color: '#ffffff'
                        }}>
                            <div style={{ fontSize: '2.5rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                                {char.character}
                            </div>
                            <div style={{ fontSize: '0.85rem', color: '#888', fontWeight: 600, marginBottom: '0.5rem' }}>
                                Score: {char.total_score.toFixed(1)}
                            </div>
                            <div style={{ width: '100%', fontSize: '0.75rem', color: '#aaa', display: 'flex', justifyContent: 'space-between' }}>
                                <span>Mean: {char.details.meaning.toFixed(1)}</span>
                                <span>Phn: {char.details.phone.toFixed(1)}</span>
                            </div>
                        </div>
                    ))}
                    {(!topCharacters || topCharacters.length === 0) && (
                        <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '2rem', color: '#666' }}>
                            No character suggestions available.
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
