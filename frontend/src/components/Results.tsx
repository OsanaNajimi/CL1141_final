'use client';
import { useState } from 'react';

// New Interfaces matching backend response
type CharacterInfo = {
    character: string;
    pinyin: string;
    meaning: string;
    absolute_gender_score: number;
    age_range: string;
    scores: {
        total_score: number;
        gender_score: number;
        age_score: number;
        meaning_score: number;
    }
};

type NameResult = {
    family_name: string;
    family_name_pinyin?: string;
    character_1: CharacterInfo;
    character_2: CharacterInfo;
    scores: {
        total_score: number;
        gender_score: number;
        age_score: number;
        meaning_score: number;
    }
};

type TopCharacter = CharacterInfo; // Backend returns same structure

// Internal Modal Component
function Modal({ isOpen, onClose, title, pinyin, children }: { isOpen: boolean, onClose: () => void, title: string, pinyin: string, children: React.ReactNode }) {
    if (!isOpen) return null;

    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            zIndex: 1000,
            backdropFilter: 'blur(5px)'
        }} onClick={onClose}>
            <div style={{
                backgroundColor: '#222',
                padding: '2rem',
                borderRadius: '12px',
                width: '90%',
                maxWidth: '500px',
                color: 'white',
                border: '1px solid #444',
                position: 'relative',
                boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.1)'
            }} onClick={e => e.stopPropagation()}>
                <button
                    onClick={onClose}
                    style={{
                        position: 'absolute',
                        top: '1rem',
                        right: '1rem',
                        background: 'none',
                        border: 'none',
                        color: '#666',
                        fontSize: '1.5rem',
                        cursor: 'pointer'
                    }}
                >
                    &times;
                </button>

                <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
                    <h2 style={{ fontSize: '3rem', margin: '0', fontFamily: 'serif' }}>{title}</h2>
                    <p style={{ fontSize: '1.2rem', color: '#888', marginTop: '0.5rem', fontStyle: 'italic' }}>{pinyin}</p>
                </div>

                <div style={{ lineHeight: '1.6', color: '#ccc' }}>
                    {children}
                </div>
            </div>
        </div>
    );
}

export default function Results({ results, topCharacters }: { results: NameResult[], topCharacters: TopCharacter[] }) {
    const [activeTab, setActiveTab] = useState<'names' | 'characters'>('names');
    const [selectedName, setSelectedName] = useState<NameResult | null>(null);
    const [selectedChar, setSelectedChar] = useState<TopCharacter | null>(null);

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
                    {results.map((item, idx) => {
                        const fullName = item.family_name + item.character_1.character + item.character_2.character;
                        // Construct generic pinyin string
                        const fullPinyin = [
                            item.family_name_pinyin,
                            item.character_1.pinyin,
                            item.character_2.pinyin
                        ].filter(Boolean).join(' ');

                        return (
                            <div key={idx}
                                onClick={() => setSelectedName(item)}
                                style={{
                                    border: '1px solid #333',
                                    borderRadius: '8px',
                                    backgroundColor: '#1a1a1a', // Dark card bg
                                    padding: '1.5rem',
                                    display: 'flex',
                                    flexDirection: 'column',
                                    boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
                                    color: '#ffffff',
                                    cursor: 'pointer',
                                    transition: 'transform 0.1s',
                                }}
                                onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
                                onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
                            >
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.5rem' }}>
                                    <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>
                                        {fullName}
                                    </div>
                                    <div style={{ fontSize: '0.85rem', color: '#888', fontWeight: 600 }}>
                                        Score: {item.scores.total_score.toFixed(1)}
                                    </div>
                                </div>
                                <div style={{ fontSize: '1rem', color: '#888', fontStyle: 'italic', marginBottom: '1rem' }}>
                                    {fullPinyin}
                                </div>

                                <div style={{ flex: 1 }}>
                                    {/* Breakdown */}
                                    <div style={{ fontSize: '0.8rem', color: '#aaa', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                            <span>Meaning Score:</span>
                                            <span>{item.scores.meaning_score.toFixed(2)}</span>
                                        </div>
                                        {/* Brief meaning preview could go here if generic */}
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            {activeTab === 'characters' && (
                <div style={{ width: '100%', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '1rem' }}>
                    {topCharacters && topCharacters.map((char, idx) => (
                        <div key={idx}
                            onClick={() => setSelectedChar(char)}
                            style={{
                                border: '1px solid #333',
                                borderRadius: '8px',
                                backgroundColor: '#1a1a1a',
                                padding: '1rem',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                textAlign: 'center',
                                color: '#ffffff',
                                cursor: 'pointer',
                                transition: 'transform 0.1s',
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
                            onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
                        >
                            <div style={{ fontSize: '2.5rem', fontWeight: 'bold', marginBottom: '0.25rem' }}>
                                {char.character}
                            </div>
                            <div style={{ fontSize: '0.9rem', color: '#888', fontStyle: 'italic', marginBottom: '0.5rem' }}>
                                {char.pinyin}
                            </div>
                            <div style={{ fontSize: '0.85rem', color: '#666', fontWeight: 600, marginBottom: '0.5rem' }}>
                                Score: {char.scores.total_score.toFixed(1)}
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

            {/* Modals */}
            <Modal
                isOpen={!!selectedName}
                onClose={() => setSelectedName(null)}
                title={selectedName ? selectedName.family_name + selectedName.character_1.character + selectedName.character_2.character : ''}
                pinyin={selectedName ? [selectedName.family_name_pinyin, selectedName.character_1.pinyin, selectedName.character_2.pinyin].join(' ') : ''}
            >
                {selectedName && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div>
                            <strong style={{ color: 'white', display: 'block', marginBottom: '0.5rem' }}>Meanings</strong>
                            <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '0.5rem 1rem', fontSize: '0.95rem' }}>
                                <span style={{ color: '#888' }}>{selectedName.character_1.character}:</span>
                                <span>{selectedName.character_1.meaning}</span>
                                <span style={{ color: '#888' }}>{selectedName.character_2.character}:</span>
                                <span>{selectedName.character_2.meaning}</span>
                            </div>
                        </div>

                        <div style={{ borderTop: '1px solid #333', paddingTop: '1rem', marginTop: '0.5rem' }}>
                            <strong style={{ color: 'white', display: 'block', marginBottom: '0.5rem' }}>Scores</strong>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                                <div style={{ background: '#333', padding: '0.5rem', borderRadius: '4px', textAlign: 'center' }}>
                                    <div style={{ fontSize: '0.8rem', color: '#888' }}>Total</div>
                                    <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{selectedName.scores.total_score.toFixed(1)}</div>
                                </div>
                                <div style={{ background: '#333', padding: '0.5rem', borderRadius: '4px', textAlign: 'center' }}>
                                    <div style={{ fontSize: '0.8rem', color: '#888' }}>Gender Bias</div>
                                    <div style={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#666', marginBottom: '0.2rem' }}>
                                            <span>Female</span>
                                            <span>Male</span>
                                        </div>
                                        <input
                                            type="range"
                                            min="0"
                                            max="1"
                                            step="0.01"
                                            value={(selectedName.character_1.absolute_gender_score + selectedName.character_2.absolute_gender_score) / 2} // Average for visualization? Or sum?
                                            // Request: "range is 0~1, 0 for female and 1 for male"
                                            // The backend sends normalized 0-1 scores for chars.
                                            // Ideally we show the bias of the *combination*?
                                            // Or just clamp the sum? 
                                            // Let's assume average of the two chars' absolute gender score helps visualize the NAME's overall gender lean. 
                                            // Backend: absolute_gender_score is 0(F)-1(M).
                                            disabled
                                            style={{ width: '100%', accentColor: 'white', cursor: 'default' }}
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </Modal>

            <Modal
                isOpen={!!selectedChar}
                onClose={() => setSelectedChar(null)}
                title={selectedChar?.character || ''}
                pinyin={selectedChar?.pinyin || ''}
            >
                {selectedChar && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div>
                            <strong style={{ color: 'white', display: 'block', marginBottom: '0.5rem' }}>Meaning</strong>
                            <p style={{ margin: 0 }}>{selectedChar.meaning}</p>
                        </div>

                        <div style={{ borderTop: '1px solid #333', paddingTop: '1rem', marginTop: '0.5rem' }}>
                            <strong style={{ color: 'white', display: 'block', marginBottom: '0.5rem' }}>Scores</strong>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                                <div style={{ background: '#333', padding: '0.5rem', borderRadius: '4px', textAlign: 'center' }}>
                                    <div style={{ fontSize: '0.8rem', color: '#888' }}>Total</div>
                                    <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{selectedChar.scores.total_score.toFixed(1)}</div>
                                </div>
                                <div style={{ background: '#333', padding: '0.5rem', borderRadius: '4px', textAlign: 'center' }}>
                                    <div style={{ fontSize: '0.8rem', color: '#888' }}>Gender Bias</div>
                                    <div style={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#666', marginBottom: '0.2rem' }}>
                                            <span>Female</span>
                                            <span>Male</span>
                                        </div>
                                        <input
                                            type="range"
                                            min="0"
                                            max="1"
                                            step="0.01"
                                            value={selectedChar.absolute_gender_score}
                                            disabled
                                            style={{ width: '100%', accentColor: 'white', cursor: 'default' }}
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.5rem', textAlign: 'center' }}>
                            Age Range: {selectedChar.age_range}
                        </div>
                    </div>
                )}
            </Modal>
        </div>
    );
}
