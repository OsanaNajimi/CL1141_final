'use client';
import { useState } from 'react';

type FormData = {
    english_name: string;
    gender: string;
    birth_date: string;
    description: string;
    surname_mode?: string;
    surname_input?: string;
    // Dynamic factors
    weight_factors: {
        meaning: number;
        gender: number;
        age: number;
        phone: number;
    };
    temperature_factor: number;
};

// Internal types for form state (0-10 sliders)
type FormState = {
    english_name: string;
    gender: string;
    birth_date: string;
    description: string;
    // Sliders 0-10 (default 5)
    w_meaning: number;
    w_gender: number;
    w_age: number;
    w_phone: number;
    w_temp: number;
};

const MEANING_PRESETS = [
    "Brave", "Beauty", "Intelligent", "Peaceful", "Creative", "Strong", "Kind", "Happy"
];

export default function InputForm({ onSubmit, isLoading }: { onSubmit: (data: FormData) => void, isLoading: boolean }) {
    const [formState, setFormState] = useState<FormState>({
        english_name: '',
        gender: 'male',
        birth_date: '1990-01-01',
        description: '',
        w_meaning: 5,
        w_gender: 5,
        w_age: 5,
        w_phone: 5,
        w_temp: 5
    });

    const [showAdvanced, setShowAdvanced] = useState(false);
    const [nameError, setNameError] = useState('');

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
        const { name, value, type } = e.target;

        if (name === 'english_name' || name === 'surname_input') {
            // Validate: check if value contains only English letters and spaces
            const isValid = /^[A-Za-z\s]*$/.test(value);
            if (!isValid) {
                return;
            }
        }

        setFormState(prev => ({
            ...prev,
            [name]: type === 'range' ? parseFloat(value) : value
        }));
    };

    const handlePresetClick = (word: string) => {
        setFormState(prev => {
            // Append? Replace? "clicking on these buttons will set the content of the input box to the text on the button"
            // "Set the content" usually means replace or add. Often user wants to combine.
            // However request says "set the content...to the text". I will implement strictly "set", 
            // but "Add to" is usually friendlier. Let's assume replace or simple append?
            // "set the content... to the text" -> Replace.
            // But if I want "Brave AND Intelligent"? 
            // I'll implement append with comma if text exists to be helpful, or just Replace if strictly following prompt.
            // Let's do Append for better UX, or just set. "Set" implies replace. I'll do replace for exact compliance but maybe append is better?
            // I'll do: if empty set, else append ", " + text.
            const current = prev.description.trim();
            const newToken = word;
            if (!current) return { ...prev, description: newToken };
            if (current.includes(newToken)) return prev; // Avoid dupe
            return { ...prev, description: current + ", " + newToken };
        });
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();

        // Final Validation check (e.g. if pasted)
        if (!/^[A-Za-z\s]+$/.test(formState.english_name)) {
            setNameError("English Name must contain only letters.");
            return;
        }
        setNameError('');

        // Prepare Payload
        // div by 5 logic
        const payload: any = { // Relaxed type for new fields
            english_name: formState.english_name,
            gender: formState.gender,
            birth_date: formState.birth_date,
            description: formState.description,
            surname_mode: (formState as any).surname_mode || 'random',
            surname_input: (formState as any).surname_input || '',
            weight_factors: {
                meaning: formState.w_meaning / 5.0,
                gender: formState.w_gender / 5.0,
                age: formState.w_age / 5.0,
                phone: formState.w_phone / 5.0
            },
            temperature_factor: formState.w_temp / 5.0
        };

        onSubmit(payload);
    };

    const inputStyle = {
        width: '100%',
        padding: '0.8rem',
        border: '1px solid #333',
        borderRadius: '4px',
        fontSize: '0.9rem',
        backgroundColor: '#1a1a1a',
        color: '#ffffff'
    };

    const labelStyle = {
        display: 'block',
        marginBottom: '0.4rem',
        fontSize: '0.8rem',
        fontWeight: 600,
        textTransform: 'uppercase' as const,
        letterSpacing: '0.05em',
        color: '#aaa'
    };

    // Helper for slider
    const renderSlider = (label: string, name: keyof FormState, min: number = 0, max: number = 10) => (
        <div style={{ marginBottom: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                <label style={labelStyle}>{label}</label>
                <span style={{ fontSize: '0.8rem', color: '#fff' }}>{formState[name]}</span>
            </div>
            <input
                type="range"
                name={name}
                min={min}
                max={max}
                step="0.5"
                value={formState[name]}
                onChange={handleChange}
                style={{ width: '100%', cursor: 'pointer', accentColor: 'white' }}
            />
        </div>
    );

    return (
        <form onSubmit={handleSubmit} style={{ width: '100%' }}>
            <div style={{ marginBottom: '1rem' }}>
                <label style={labelStyle}>English Name</label>
                <input
                    name="english_name"
                    type="text"
                    required
                    style={{ ...inputStyle, borderColor: nameError ? 'red' : '#333' }}
                    value={formState.english_name}
                    onChange={handleChange}
                    placeholder="e.g. John"
                />
                {nameError && <div style={{ color: 'red', fontSize: '0.8rem', marginTop: '0.2rem' }}>{nameError}</div>}
            </div>

            <div style={{ marginBottom: '1rem' }}>
                <label style={labelStyle}>Gender</label>
                <select
                    name="gender"
                    style={inputStyle}
                    value={formState.gender}
                    onChange={handleChange}
                >
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="neutral">Neutral</option>
                </select>
            </div>

            {/* Surname Options */}
            <div style={{ marginBottom: '1rem' }}>
                <label style={labelStyle}>Family Name Mode</label>
                <select
                    name="surname_mode"
                    style={inputStyle}
                    value={(formState as any).surname_mode || 'none'}
                    onChange={(e) => setFormState(prev => ({ ...prev, surname_mode: e.target.value }))}
                >
                    <option value="none">None</option>
                    <option value="fixed">Specify Chinese Surname</option>
                    <option value="phonetic">Match English Surname (Phonetic)</option>
                    <option value="random">Random</option>
                </select>
            </div>

            {/* Conditional Surname Input */}
            {((formState as any).surname_mode === 'fixed' || (formState as any).surname_mode === 'phonetic') && (
                <div style={{ marginBottom: '1rem' }}>
                    <label style={labelStyle}>
                        {(formState as any).surname_mode === 'fixed' ? 'Chinese Surname' : 'English Surname'}
                    </label>
                    <input
                        name="surname_input"
                        type="text"
                        style={inputStyle}
                        value={(formState as any).surname_input || ''}
                        onChange={(e) => setFormState(prev => ({ ...prev, surname_input: e.target.value }))}
                        placeholder={(formState as any).surname_mode === 'fixed' ? 'e.g. 李' : 'e.g. Martin'}
                    />
                </div>
            )}

            <div style={{ marginBottom: '1rem' }}>
                <label style={labelStyle}>Date of Birth</label>
                <input
                    name="birth_date"
                    type="date"
                    required
                    style={inputStyle}
                    value={formState.birth_date}
                    onChange={handleChange}
                />
            </div>

            <div style={{ marginBottom: '1rem' }}>
                <label style={labelStyle}>Meaning Description</label>

                {/* Presets */}
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '0.5rem' }}>
                    {MEANING_PRESETS.map(word => (
                        <button
                            key={word}
                            type="button"
                            onClick={() => handlePresetClick(word)}
                            style={{
                                padding: '0.3rem 0.6rem',
                                fontSize: '0.75rem',
                                border: '1px solid #444',
                                borderRadius: '15px',
                                background: 'transparent',
                                color: '#ccc',
                                cursor: 'pointer',
                                transition: 'all 0.2s'
                            }}
                            onMouseOver={e => { e.currentTarget.style.borderColor = '#888'; e.currentTarget.style.color = 'white'; }}
                            onMouseOut={e => { e.currentTarget.style.borderColor = '#444'; e.currentTarget.style.color = '#ccc'; }}
                        >
                            {word}
                        </button>
                    ))}
                </div>

                <textarea
                    name="description"
                    required
                    rows={3}
                    style={{ ...inputStyle, resize: 'vertical' }}
                    value={formState.description}
                    onChange={handleChange}
                    placeholder="e.g. Brave, Intelligent..."
                />
            </div>

            {/* Advanced Toggle */}
            <div style={{ marginBottom: '1rem' }}>
                <button
                    type="button"
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    style={{
                        background: 'none',
                        border: 'none',
                        color: '#888',
                        fontSize: '0.85rem',
                        cursor: 'pointer',
                        textDecoration: 'underline',
                        padding: 0
                    }}
                >
                    {showAdvanced ? "Hide Advanced Options ▲" : "Show Advanced Options ▼"}
                </button>
            </div>

            {/* Advanced Options Area */}
            {showAdvanced && (
                <div style={{
                    padding: '1rem',
                    borderRadius: '8px',
                    backgroundColor: '#222',
                    marginBottom: '1rem',
                    border: '1px solid #333',
                    animation: 'fadeIn 0.3s'
                }}>
                    <h4 style={{ margin: '0 0 1rem 0', fontSize: '0.9rem', color: '#fff' }}>Weights Adjustment (0 - 10)</h4>

                    {renderSlider("Meaning Weight", 'w_meaning')}
                    {renderSlider("Gender Weight", 'w_gender')}
                    {renderSlider("Age Weight", 'w_age')}
                    {renderSlider("Phonetic Weight", 'w_phone')}

                    <div style={{ borderTop: '1px solid #444', margin: '1rem 0' }}></div>

                    {renderSlider("Creativity (Temperature)", 'w_temp', 0, 10)}
                    <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '-0.5rem' }}>
                        Higher temperature = More random/creative names.
                    </div>
                </div>
            )}

            <button
                type="submit"
                disabled={isLoading}
                style={{
                    width: '100%',
                    padding: '1rem',
                    backgroundColor: 'white',
                    color: 'black',
                    border: 'none',
                    borderRadius: '4px',
                    fontSize: '0.9rem',
                    fontWeight: 'bold',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                    cursor: isLoading ? 'not-allowed' : 'pointer',
                    opacity: isLoading ? 0.7 : 1,
                    marginTop: '0.5rem'
                }}
            >
                {isLoading ? 'GENERATING...' : 'GET A NAME'}
            </button>
        </form>
    );
}
