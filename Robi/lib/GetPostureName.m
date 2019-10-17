function [posture] = GetPostureName(postureName)
    if postureName == 'sit_long'
        posture = 'Sit Long';
    elseif postureName == 'sit_medium'
        posture = 'Sit Medium';
    elseif postureName == 'sit_short'
        posture = 'Sit Short';
    elseif postureName ==  'stand_long'
        posture = 'Stand Long';
    elseif postureName == 'stand_medium'
        posture = 'Stand Medium';
    else
        posture = 'Stand Short';
    end
end

