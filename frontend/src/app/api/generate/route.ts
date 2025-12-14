import { NextResponse } from 'next/server';

export async function POST(req: Request) {
    try {
        const body = await req.json();
        // Use environment variable for backend URL, default to localhost for dev
        const backendBaseUrl = process.env.BACKEND_URL || 'http://localhost:5000';
        const flaskUrl = `${backendBaseUrl}/api/generate_names`;

        console.log(`Proxying request to ${flaskUrl}`);

        const response = await fetch(flaskUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`Flask API responded with ${response.status}: ${errorText}`);
            return NextResponse.json(
                { error: `Backend Error: ${response.status}`, details: errorText },
                { status: response.status }
            );
        }

        const data = await response.json();
        return NextResponse.json(data);

    } catch (error) {
        console.error('API Proxy Error:', error);
        return NextResponse.json(
            {
                error: 'Failed to connect to backend generator.',
                details: 'Make sure generate_names.py is running in WSL2 (port 5000).'
            },
            { status: 500 }
        );
    }
}
