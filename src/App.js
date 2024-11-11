 import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function Chatbot() {
  const [messages, setMessages] = useState([
    { text: 'Hi there! Welcome to Sac State!', sender: 'bot', showEmojis: false },
    { text: 'How can I assist you today?', sender: 'bot', showEmojis: false }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState([]);  // Added state for suggestions
  const [showFeedback, setShowFeedback] = useState(false);
  const [rating, setRating] = useState(0);
  const [showTourOptions, setShowTourOptions] = useState(false);
  const [showTuitionOptions, setShowTuitionOptions] = useState(false);
  const [showActivitesOptions, setShowActivitesOptions] = useState(false);
  const [showOnCampusJobOptions, setShowOnCampusJobOptions] = useState(false);
  const [editingMessageIndex, setEditingMessageIndex] = useState(null);
  const [isListening, setIsListening] = useState(false);
  const [emojiFeedback, setEmojiFeedback] = useState({}); // Store emoji feedback for each bot response
  const [showEmojiPopup, setShowEmojiPopup] = useState(false); // For the emoji popup control
  const [currentBotMessageIndex, setCurrentBotMessageIndex] = useState(null); // Track current bot message for emoji popup

  const recognitionRef = useRef(null); // Store recognition across renders

  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      recognitionRef.current = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onstart = () => {
        setIsListening(true);
      };

      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInputMessage(transcript); // Set the recognized speech as input
        setIsListening(false); // Stop listening after receiving the result
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };
    }
  }, []);

  // Function to toggle speech recognition
  const handleMicClick = () => {
    if (isListening) {
      recognitionRef.current.stop(); // Stop recording
    } else {
      recognitionRef.current.start(); // Start recording
    }
  };

  // Handle sending the user's message
  const handleSendMessage = async () => {
    if (inputMessage.trim() !== '') {
      if (editingMessageIndex !== null) {
        const updatedMessages = [...messages];
        updatedMessages[editingMessageIndex] = { text: inputMessage, sender: 'user' };
        setMessages(updatedMessages);
        setEditingMessageIndex(null); // Exit edit mode
      } else {
        setMessages([...messages, { text: inputMessage, sender: 'user' }]);
      }

      if (inputMessage.toLowerCase() === 'exit') {
        setShowFeedback(true);
      } else {
        await sendMessageToBackend(inputMessage);
      }
      setInputMessage('');
    }
  };

  const sendMessageToBackend = async (message) => {
    setLoading(true);
    setSuggestions([]);  // Clear previous suggestions when sending a new message

    try {
      const response = await fetch('http://127.0.0.1:5001/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: message }),
      });

      const data = await response.json();

      if (data.type === 'image') {
        setMessages((messages) => [
          ...messages,
          { text: data.response, sender: 'bot', isImage: true, showEmojis: true }
        ]);
      } else {
        setMessages((messages) => [
          ...messages,
          { text: data.response, sender: 'bot', isImage: false, showEmojis: true }
        ]);
      }

      setSuggestions(data.suggestions || []);  // Set the suggestions from the backend

      setCurrentBotMessageIndex(messages.length); // Set the index of the new bot message for emoji popup
      setShowEmojiPopup(true); // Trigger the emoji popup
    } catch (error) {
      console.error('Error:', error);
      setMessages((messages) => [
        ...messages,
        { text: 'Sorry, something went wrong.', sender: 'bot', isImage: false, showEmojis: true }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleStarClick = (stars) => {
    setRating(stars);
  };

  const submitFeedback = async () => {
    try {
      const feedbackData = { response_text: 'feedback', feedback: rating };
      await fetch('http://127.0.0.1:5001/submit_feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedbackData),
      });
      alert('Thank you for your feedback!');
      setShowFeedback(false);
      setRating(0);
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  // Define missing functions
  const handleTourClick = () => setShowTourOptions(!showTourOptions);
  const handleTuitionClick = () => setShowTuitionOptions(!showTuitionOptions);
  const handleActivitesClick = () => setShowActivitesOptions(!showActivitesOptions);
  const handleOnCampusJobClick = () => setShowOnCampusJobOptions(!showOnCampusJobOptions);
  const handleVirtualTourClick = () => window.open('https://www.youvisit.com/tour/60139/79933/', '_blank');
  const handleYoutubeClick = () => window.open('https://youtu.be/JEqv--3D_M8?si=XcUo8VlhErnUEh9j', '_blank');
  const handleScheduleClick = () => window.open('https://s.visitdays.com/csusao/ci/cjcskhncfv', '_blank');
  const handleFeeClick = () => window.open('https://www.csus.edu/administration-business-affairs/bursar/tuition-living-costs.html', '_blank');
  const handlePaymentClick = () => window.open('https://www.csus.edu/administration-business-affairs/bursar/payment.html', '_blank');
  const handleStudentClubsClick = () => window.open('https://www.csus.edu/college/arts-letters/clubs-organizations-arts-letters.html', '_blank');
  const handleUpComingClick = () => window.open('https://events.csus.edu/?filterview=FeaturedEvents', '_blank');

  // Function to handle emoji click for feedback
  const handleEmojiClick = (emoji) => {
    setEmojiFeedback((prevFeedback) => ({
      ...prevFeedback,
      [currentBotMessageIndex]: emoji, // Store emoji feedback for the current bot message
    }));
    setShowEmojiPopup(false); // Close emoji popup after selection
  };

  const handleEditMessage = (messageText, index) => {
    setInputMessage(messageText);
    setEditingMessageIndex(index);
  };

  // Function to render bot response text with clickable links if the message contains URLs or email addresses
  const renderBotResponseWithLinks = (text) => {
    const urlRegex = /(https?:\/\/[^\s]+)/g; // Regular expression to identify URLs
    const emailRegex = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b/; // Regular expression to identify emails
    const parts = text.split(' ');

    return parts.map((part, index) => {
      if (urlRegex.test(part)) {
        return (
          <a key={index} href={part} target="_blank" rel="noopener noreferrer" style={{ color: 'blue' }}>
            {part}
          </a>
        );
      } else if (emailRegex.test(part)) {
        const gmailLink = `https://mail.google.com/mail/?view=cm&fs=1&to=${part}`;
        const outlookLink = `https://outlook.office.com/mail/deeplink/compose?to=${part}`;
        return (
          <span key={index}>
            <a href={gmailLink} target="_blank" rel="noopener noreferrer" style={{ color: 'blue', marginRight: '10px' }}>
              Gmail
            </a>
            <a href={outlookLink} target="_blank" rel="noopener noreferrer" style={{ color: 'blue' }}>
              Outlook
            </a>
          </span>
        );
      } else {
        return part + ' ';
      }
    });
  };

  const handleSuggestionClick = (suggestion) => {
    setInputMessage(suggestion);  // Set suggestion as the next input message
  };

  return (
    <div className="chat-container">
      <div className="header">
        <img 
          src="https://pbs.twimg.com/profile_images/1694778608479522819/X2i2t1LN_400x400.jpg" 
          alt="Sac State Logo" 
          className="logo" 
        />
        <h1>Campus Connect</h1>
      </div>

      <div className="gpt-style-icons">
        <button className="icon-button" onClick={handleTourClick}>
          <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQkLFM1TI83AlEWMMt-B_GwzLB5e4n2AwGKHw&s" alt="Campus Tour" /> Campus Tour
        </button>
        <button className="icon-button" onClick={handleTuitionClick}>
          <img src="https://as2.ftcdn.net/v2/jpg/03/03/70/17/1000_F_303701714_ouxRjI3fGC5K86UY1jsEDissYqIJxAnb.jpg" alt="Tuition" /> Tuition
        </button>
        <button className="icon-button" onClick={handleActivitesClick}>
          <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRM7eXhwTx2l5mhCiXsGx1MjM1uUbcB4oQ2_w&s" alt="Activites" /> Activites
        </button>
      </div>

      {showTourOptions && (
        <div className="sub-options">
          <button className="sub-option-button" onClick={handleVirtualTourClick}>Virtual Tour</button>
          <button className="sub-option-button" onClick={handleYoutubeClick}>YouTube Video</button>
          <button className="sub-option-button" onClick={handleScheduleClick}>Schedule Appointment</button>
        </div>
      )}

      {showTuitionOptions && (
        <div className="sub-options">
          <button className="sub-option-button" onClick={handleFeeClick}>Fee Information</button>
          <button className="sub-option-button" onClick={handlePaymentClick}>Payment Information</button>
        </div>
      )}

      {showActivitesOptions && (
        <div className="sub-options">
          <button className="sub-option-button" onClick={handleStudentClubsClick}>Student Clubs</button>
          <button className="sub-option-button" onClick={handleOnCampusJobClick}>
            OnCampus Jobs
          </button>
          <button className="sub-option-button" onClick={handleUpComingClick}>UpComing Events</button>
        </div>
      )}

      {showOnCampusJobOptions && (
        <div className="sub-options">
          <button className="sub-option-button" onClick={() => window.open('https://csus.joinhandshake.com/login', '_blank')}>
            Handshake
          </button>
          <button className="sub-option-button" onClick={() => window.open('https://careers.aramark.com/', '_blank')}>
            Aramark
          </button>
          <button className="sub-option-button" onClick={() => window.open('https://www.enterprises.csus.edu/human-resources/ueiworkforce/', '_blank')}>
            UEI
          </button>
        </div>
      )}

      <div className="chat-box">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.sender === 'user' ? 'user-message' : ''}`}>
            {message.sender === 'bot' && (
              <>
                {message.isImage ? (
                  <img 
                    src={`http://127.0.0.1:5001${message.text}`} 
                    alt="Bot Response" 
                    className="response-image"
                    style={{
                      width: '100%',
                      height: 'auto',
                      maxWidth: '300px',
                      objectFit: 'cover',
                      padding: '10px',
                      borderRadius: '8px'
                    }}
                  />
                ) : (
                  <p>{renderBotResponseWithLinks(message.text)}</p>
                )}
                {/* Show feedback selection */}
                {emojiFeedback[index] && <p>You selected: {emojiFeedback[index]}</p>}
              </>
            )}
            {message.sender === 'user' && (
              <div className="user-message-content">
                <p>{`You: ${message.text}`}</p>
                <img
                  src="https://img.icons8.com/ios-glyphs/30/000000/edit--v1.png"
                  alt="Edit"
                  className="edit-icon"
                  onClick={() => handleEditMessage(message.text, index)} 
                  style={{ cursor: 'pointer', marginLeft: '10px' }}
                />
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="loading">Bot is typing...</div>
        )}
      </div>

      {/* Sliding Suggestions Section */}
      {suggestions.length > 0 && (
        <div className="suggestions-box">
          <h3>Suggestions:</h3>
          <div className="suggestions-slider">
            {suggestions.map((suggestion, index) => (
              <div 
                key={index} 
                className="suggestion-slide" 
                onClick={() => handleSuggestionClick(suggestion)}
                title={suggestion}  // This will show the full suggestion on hover
              >
                {suggestion.length > 20 ? suggestion.substring(0, 20) + '...' : suggestion}  {/* Truncate suggestion */}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Modal-like emoji pop-up */}
      {showEmojiPopup && (
        <div className="emoji-popup-overlay">
          <div className="emoji-popup">
            <div className="emoji-options">
              <span onClick={() => handleEmojiClick('😊')} style={{ cursor: 'pointer', marginRight: '4px', fontSize: '1rem' }}>😊</span>
              <span onClick={() => handleEmojiClick('😢')} style={{ cursor: 'pointer', marginRight: '4px', fontSize: '1rem' }}>😢</span>
              <span onClick={() => handleEmojiClick('😐')} style={{ cursor: 'pointer', marginRight: '4px', fontSize: '1rem' }}>😐</span>
              <span onClick={() => handleEmojiClick('😡')} style={{ cursor: 'pointer', marginRight: '4px', fontSize: '1rem' }}>😡</span>
            </div>
          </div>
        </div>
      )}

      <div className="input-box">
        <input 
          type="text" 
          value={inputMessage} 
          onChange={(e) => setInputMessage(e.target.value)} 
          placeholder="Type a message" 
        />
        <button onClick={handleSendMessage}>{editingMessageIndex !== null ? 'Update' : 'Send'}</button>
        <button 
          onClick={handleMicClick} 
          style={{
            backgroundColor: 'white', 
            border: 'none', 
            borderRadius: '50%', 
            padding: '10px', 
            cursor: 'pointer',
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          <img 
            src={isListening ? "https://img.icons8.com/ios-filled/50/000000/microphone.png" : "https://cdn1.iconfinder.com/data/icons/social-messaging-ui-color/254000/79-512.png"} 
            alt="Mic"
            style={{ width: '30px', height: '30px' }}
          />
        </button>
      </div>

      {showFeedback && (
        <div className="feedback-section">
          <h3>Please rate your experience:</h3>
          <div className="stars">
            {[1, 2, 3, 4, 5].map(star => (
              <span 
                key={star} 
                className={`star ${star <= rating ? 'selected' : ''}`} 
                onClick={() => handleStarClick(star)}
              >
                ★
              </span>
            ))}
          </div>
          <button onClick={submitFeedback}>Submit Feedback</button>
        </div>
      )}
    </div>
  );
}

export default Chatbot;