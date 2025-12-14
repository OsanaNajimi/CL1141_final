'use client';
import { useState } from 'react';

type FormData = {
    english_name: string;
    gender: string;
    birth_date: string;
    description: string;
};

export default function InputForm({ onSubmit, isLoading }: { onSubmit: (data: FormData) => void, isLoading: boolean }) {
    const [formData, setFormData] = useState<FormData>({
        english_name: '',
        gender: 'male',
        birth_date: '1990-01-01',
        description: ''
    });

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        onSubmit(formData);
    };

    const inputStyle = {
        width: '100%',
        padding: '0.8rem',
        marginBottom: '1rem',
        border: '1px solid #333', // Dark border
        borderRadius: '4px',
        fontSize: '0.9rem',
        backgroundColor: '#1a1a1a', // Dark input bg
        color: '#ffffff' // Light text
    };

    const labelStyle = {
        display: 'block',
        marginBottom: '0.4rem',
        fontSize: '0.8rem',
        fontWeight: 600,
        textTransform: 'uppercase' as const,
        letterSpacing: '0.05em',
        color: '#aaa' // Lighter label text
    };

    return (
        <form onSubmit={handleSubmit} style={{ width: '100%' }}>
            <div>
                <label style={labelStyle}>English Name</label>
                <input
                    name="english_name"
                    type="text"
                    required
                    style={inputStyle}
                    value={formData.english_name}
                    onChange={handleChange}
                    placeholder="e.g. John"
                />
            </div>

            <div>
                <label style={labelStyle}>Gender</label>
                <select
                    name="gender"
                    style={inputStyle}
                    value={formData.gender}
                    onChange={handleChange}
                >
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="neutral">Neutral</option>
                </select>
            </div>

            <div>
                <label style={labelStyle}>Date of Birth</label>
                <input
                    name="birth_date"
                    type="date"
                    required
                    style={inputStyle}
                    value={formData.birth_date}
                    onChange={handleChange}
                />
            </div>

            <div>
                <label style={labelStyle}>Meaning Description</label>
                <textarea
                    name="description"
                    required
                    rows={4}
                    style={{ ...inputStyle, resize: 'vertical' }}
                    value={formData.description}
                    onChange={handleChange}
                    placeholder="e.g. Brave, Intelligent, Peaceful..."
                />
            </div>

            <button
                type="submit"
                disabled={isLoading}
                style={{
                    width: '100%',
                    padding: '1rem',
                    backgroundColor: 'white', // High contrast button
                    color: 'black',
                    border: 'none',
                    borderRadius: '4px',
                    fontSize: '0.9rem',
                    fontWeight: 'bold',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                    cursor: isLoading ? 'not-allowed' : 'pointer',
                    opacity: isLoading ? 0.7 : 1,
                    marginTop: '1rem'
                }}
            >
                {isLoading ? 'GENERATING...' : 'GET A NAME'}
            </button>
        </form>
    );
}
